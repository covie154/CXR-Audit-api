#%%
# Initial settings & imports
##############################
### LLM CXR GRADING SCRIPT ###
##############################

'''
Grading (R):
1 = normal without any findings
2 = normal variant or minor pathology, does not require follow up
3 = abnormal, non-urgent follow-up required
4 = abnormal, potentially important finding
5 = critical, urgent follow-up required
'''

from openai import OpenAI
import sys
import json
import pandas as pd
import numpy as np
import os

# Try importing prompts with fallback
# I honestly don't know why this doesn't work properly
try:
    import prompts
except ModuleNotFoundError:
    # If running in interactive mode, try different paths
    sys.path.append('.')
    sys.path.append('./cxr_audit')
    import prompts

#from llm_iter import loopEvaluate    
from helpers import levenshtein_distance, closestFinding, encode_findings, parse_list_dict, getLLMJSON

class CXRClassifier():
    """
    Class for CXR grading using LLMs.
    This class handles the grading of chest X-ray reports using a semi-algorithmic approach,
    LLM-based grading, and hybrid grading methods.
    """

    def __init__(self, findings, tubes_lines, diagnoses, model_name="o4-mini", base_url=None, api_key=None, llm=None, log_level=0):
        self.findings = findings
        self.tubes_lines = tubes_lines
        self.diagnoses = diagnoses
        self.model_name = model_name
        
        # Initialize OpenAI clients
        client_kwargs = {}
        if base_url:
            client_kwargs['base_url'] = base_url
        if api_key:
            client_kwargs['api_key'] = api_key
            
        if llm:
            self.client = llm
        else:
            self.client = OpenAI(**client_kwargs)

        ### TODO: Implement Log level
        self.log_level = log_level

        padchest_findings = self.findings.keys()
        self.padchest_findings_str = ", ".join(padchest_findings)

        tubes_lines_findings = self.tubes_lines.keys()
        self.tubes_lines_findings_str = ", ".join(tubes_lines_findings)

        diagnoses_findings = self.diagnoses.keys()
        self.diagnoses_str = ", ".join(diagnoses_findings)
    
    def create_chat_message(self, role, content):
        """Helper method to create chat messages in OpenAI format"""
        return {"role": role, "content": content}
    
    #LLM System Role Message
    @property
    def sys_role_msg(self):
        return prompts.sys_role_msg
    
    @property
    def sys_role_chat_msg(self):
        return self.create_chat_message(role="developer", content=self.sys_role_msg)

    ##################################
    ### METHOD 1: SEMI-ALGORITHMIC ###
    ##################################

    def semanticExtractionCXR(self, rpt):
        """
        Semialgorithmic semantic extraction task for a chest X-ray (CXR) report.
        This function processes the given report text through the defined LLM to
        extract findings, medical devices and diagnoses mentioned in the report.

        This is part of the get_findings_and_lines() function, which is used for
        the final processing.

        Args:
            rpt (str): The chest X-ray report text to analyze

        Returns:
            dict: Encoded findings and devices information extracted from the report
            (schema: PositiveFindings).
        """
        subs = {
            'rpt': rpt,
            'padchest_findings_str': self.padchest_findings_str,
            'tubes_lines_findings_str': self.tubes_lines_findings_str,
            'diagnoses_str': self.diagnoses_str
        }
        prompt_findings_chat_msg = self.create_chat_message(role="user", content=prompts.prompt_get_findings.substitute(**subs))
        messages_lst = [self.sys_role_chat_msg, prompt_findings_chat_msg]
        findings_devices = getLLMJSON(messages_lst, self.client, self.model_name, prompts.PositiveFindings)
        findings_devices_encoded = encode_findings(findings_devices, self.findings, self.tubes_lines, self.diagnoses)

        return findings_devices_encoded

    def get_findings_and_lines(self, rpt):
        """
        Extract findings, medical devices/lines, and diagnoses from a CXR report.
        The report first passes through the semantic extraction function to derive the three
        categories above (using the dictonaries padchest_findings, tubes_lines_findings, diagnoses)
        and derives information about the finding, certainty (if applicable) and
        temporal information (if applicable).
        
        The results of this function should be passed to get_priorities() below.

        Args:
            rpt (str): The chest X-ray report text to analyze.

        Returns: 
        dict: A dict with three components:
            - report_findings: List of dictionaries containing detected medical findings
            - report_lines: List of dictionaries containing detected medical devices/lines
            - report_diagnoses: List of dictionaries containing detected diagnoses
            Each component may be None if nothing is detected.

        Notes:
            The function prints debug information to the console showing the extracted information.
        """

        result = self.semanticExtractionCXR(rpt)

        if 'findings_all' in result and result['findings_all']:
            findings_list = []
            for finding in result['findings_all']:
                one_finding = finding['finding']
                if isinstance(one_finding, int):
                    finding_text = list(self.findings)[one_finding]
                    findings_list.append(f"({one_finding}) {finding_text}")
                else:
                    findings_list.append(f'!{one_finding}!')
            
        if 'devices_all' in result and result['devices_all']:
            devices_list = []
            for device in result['devices_all']:
                one_device = device['medical_device']
                if isinstance(one_device, int):
                    device_text = list(self.tubes_lines)[one_device]
                    devices_list.append(f"({one_device}) {device_text}")
                else:
                    devices_list.append(f'!{one_device}!')
            
        if 'diagnoses_all' in result and result['diagnoses_all']:
            diagnoses_list = []
            for diagnosis in result['diagnoses_all']:
                one_diagnosis = diagnosis['diagnosis']
                if isinstance(one_diagnosis, int):
                    diagnosis_text = list(self.diagnoses)[one_diagnosis]
                    diagnoses_list.append(f"({one_diagnosis}) {diagnosis_text}")
                else:
                    diagnoses_list.append(f'!{one_diagnosis}!')
            
        # Debug logging
        if self.log_level == 1:
            print(f'Report: {rpt}\n')
            print('Findings:', end=' ')
            
            if findings_list:
                print('Findings:', end=' ')
                print(', '.join(findings_list))
            else:
                print("No findings detected")
                
            if devices_list:
                print('Devices:', end=' ')
                print(', '.join(devices_list))
            else:
                print("No devices detected")
                
            if diagnoses_list:
                print('Diagnoses:', end=' ')
                print(', '.join(diagnoses_list))
            else:
                print("No diagnoses detected")

            print('---')
        
        return {
            'report_findings': result['findings_all'] if 'findings_all' in result else None,
            'report_lines': result['devices_all'] if 'devices_all' in result else None,
            'report_diagnoses': result['diagnoses_all'] if 'diagnoses_all' in result else None,
        }

    # Function to extract priorities from findings and lines
    def get_priorities(self, rpt_findings, rpt_lines, rpt_diagnoses):
        """
        Calculate priority scores for radiology report findings, lines/devices, and diagnoses.
        This function processes structured report data to determine clinical priority scores
        for each finding, line/device, and diagnosis. It adjusts base priorities based on 
        temporal information (new, worse, better, stable) and uncertainty factors. The priority
        scores range from 1 to 5, where higher values indicate higher clinical importance.

        Args:
            rpt_findings (list or str): Structured data derived from get_findings_and_lines(), either empty 
                or contains three keys, 'finding', 'temporal', and 'uncertainty'.
            rpt_lines (list or str): Structured data derived from get_findings_and_lines(), either empty
                or contains two keys, 'medical_device' and 'placement'.
            rpt_diagnoses (list or str): Structured data derived from get_findings_and_lines(), either empty
                or contains two keys, 'diagnosis' and 'temporal'.

        Returns:
            dict: A dict containing the following keys:
                - findings_priorities: List of priority scores for each finding (1-5)
                - lines_priorities: List of priority scores for each line/device (1-5)
                - diagnoses_priorities: List of priority scores for each diagnosis (1-5)
                - max_finding_priority: Maximum priority among findings
                - max_line_priority: Maximum priority among lines/devices
                - max_diagnosis_priority: Maximum priority among diagnoses
                - overall_max_priority: Maximum priority across all categories
        
        Notes:
            - Priority scores range from 1-5, where 1 is lowest priority and 5 is highest
            - Empty categories default to a priority of 1
            - The function relies on external dictionaries (padchest, tubes_lines, diagnoses) 
            to determine base priority levels
            - Temporal modifiers affect priority: 'new'/'worse' increases priority, 
            'better'/'stable' decreases priority (TBC)
            - Uncertainty decreases priority
            - Malpositioned lines/devices are automatically assigned priority 5
        """

        findings_priorities = []
        lines_priorities = []
        diagnoses_priorities = []
        
        # Parse findings
        findings = parse_list_dict(rpt_findings)
        if findings:
            for finding in findings:
                finding_index = finding.get('finding')
                if isinstance(finding_index, int):
                    finding_name = list(self.findings.keys())[finding_index]
                    priority = self.findings[finding_name]

                    ### QUESTION:
                    # Should we comment this out?
                    '''
                    # Check if temporal is 'new' (0) or 'worse' (2) and increase priority by 1 if so
                    temporal = finding.get('temporal')
                    if temporal in [0, 2]:  # 0 is 'new', 2 is 'worse'
                        priority += 1
                    if temporal in [1, 3]:  # 1 is 'better', 3 is 'stable'
                        priority -= 1
                    '''

                    # Check if uncertainty is 'uncertain' (1) and decrease priority by 1
                    uncertainty = finding.get('uncertainty')
                    if uncertainty == 1:  # 1 is 'uncertain'
                        priority -= 1

                    # Ensure priority is within the range of 1 to 5
                    priority = max(1, min(priority, 5))

                    # Append the priority to the list
                    findings_priorities.append(priority)
        
        # Parse lines/devices
        lines = parse_list_dict(rpt_lines)
        if lines:
            for line in lines:
                device_index = line.get('medical_device')
                if isinstance(device_index, int):
                    device_name = list(self.tubes_lines.keys())[device_index]
                    priority = self.tubes_lines[device_name]
                    # Check if placement is 'malpositioned' (2) and change priority to 5
                    placement = line.get('placement')
                    if placement == 2:
                        priority = 5
                    
                    # Append the priority to the list
                    lines_priorities.append(priority)
        
        # Parse diagnoses
        all_dx = parse_list_dict(rpt_diagnoses)
        if all_dx:
            for diagnosis in all_dx:
                diagnosis_index = diagnosis.get('diagnosis')
                if isinstance(diagnosis_index, int):
                    diagnosis_name = list(self.diagnoses.keys())[diagnosis_index]
                    priority = self.diagnoses[diagnosis_name]

                    # Check if temporal is 'better' (1) or 'stable' (3) and decrease priority by 1 if so
                    temporal = diagnosis.get('temporal')
                    if temporal in [1, 3]:  # 1 is 'better', 3 is 'stable'
                        priority -= 1
                    
                    # Ensure priority is within the range of 1 to 5
                    priority = max(1, min(priority, 5))

                    # Append the priority to the list
                    diagnoses_priorities.append(priority)
        
        # Create a dictionary to hold the new values
        new_values = {
            'findings_priorities': findings_priorities if findings_priorities else [1],
            'lines_priorities': lines_priorities if lines_priorities else [1],
            'diagnoses_priorities': diagnoses_priorities if diagnoses_priorities else [1],
        }
        
        # Add max priorities if applicable
        # Set max_finding_priority to max of findings_priorities or 1 (normal) if empty (i.e. no findings)
        if findings_priorities:
            new_values['max_finding_priority'] = max(findings_priorities)
        else:
            new_values['max_finding_priority'] = 1
        
        # Do the same for the lines findings
        if lines_priorities:
            new_values['max_line_priority'] = max(lines_priorities)
        else:
            new_values['max_line_priority'] = 1
            
        
        if diagnoses_priorities:
            new_values['max_diagnosis_priority'] = max(diagnoses_priorities)
        else:
            new_values['max_diagnosis_priority'] = 1

        # Use findings_priorities and lines_priorities to set overall_max_priority
        # If both are empty (should not happen), set overall_max_priority to 1 
        if findings_priorities or lines_priorities:
            new_values['overall_max_priority'] = max(findings_priorities + lines_priorities + diagnoses_priorities)
        else:
            new_values['overall_max_priority'] = 1
        
        if not findings_priorities and not lines_priorities:
            new_values['overall_max_priority'] = 1
        
        return new_values

    def gradeReportSemialgo(self, rpt):
        """
        Grade a chest X-ray report using a semi-algorithmic approach.
        This function processes the report to extract findings, lines/devices, and diagnoses,
        then calculates priority scores for each category. It returns a dictionary with the
        findings, lines, diagnoses, and their respective priorities.

        Args:
            rpt (str): The chest X-ray report text to analyze.

        Returns:
            dict: A dictionary containing:
                - report_findings: List of findings with their details
                - report_lines: List of lines/devices with their details
                - report_diagnoses: List of diagnoses with their details
                - findings_priorities: List of priority scores for findings
                - lines_priorities: List of priority scores for lines/devices
                - diagnoses_priorities: List of priority scores for diagnoses
                - max_finding_priority: Maximum priority among findings
                - max_line_priority: Maximum priority among lines/devices
                - max_diagnosis_priority: Maximum priority among diagnoses
                - overall_max_priority: Maximum priority across all categories
        """
        
        # Get findings, lines and diagnoses from the report
        results = self.get_findings_and_lines(rpt)
        
        # Calculate priorities based on the extracted data
        priorities = self.get_priorities(results['report_findings'], results['report_lines'], results['report_diagnoses'])

        # Combine results and priorities into a single dictionary
        combined_results = {
            'report_findings': results['report_findings'],
            'report_lines': results['report_lines'],
            'report_diagnoses': results['report_diagnoses'],
            **priorities  # Unpacks the priorities dictionary key-value pairs into combined_results
        }
        
        return combined_results

    #############################
    ### METHOD 2: LLM GRADING ###
    #############################

    def gradeReportLLM(self, rpt):
        """
        Approach 2: Grading a CXR report using LLM approach. This function passes the report to llm
        and returns the grade of the report as a JSON object.

        Args:
            rpt (str): The chest X-ray report text to be graded.

        Returns:
            dict: A JSON object containing the grading results from the language model. (grade: int)
        """
        subs = {
            'rpt': rpt
        }
        prompt_grade_chat_msg = self.create_chat_message(role="user", content=prompts.prompt_grade_llm.substitute(subs))
        messages_lst = [self.sys_role_chat_msg, prompt_grade_chat_msg]

        return getLLMJSON(messages_lst, self.client, self.model_name, prompts.cxrGrade)
    
    ################################
    ### METHOD 3: HYBRID GRADING ###
    ################################

    def gradeReportHybrid(self, rpt, grade_semialgo):
        """
        Grades a chest x-ray report using a hybrid approach that combines 
        the results from semi-algorithmic grading (method 1) with LLM-based evaluation
        rather than separating it and then judging it.
        
        Args:
            rpt (str): The chest x-ray report to be graded.
            grade_semialgo (int): The grade from the semi-algorithmic method.

        Returns:
            dict: A JSON object containing the grading results from the LLM evaluation 
            (grade: int, explanation: str).
        """
        subs = {
            'rpt': rpt,
            'grade_semialgo': grade_semialgo,
            'padchest_findings_str': self.padchest_findings_str,
            'diagnoses_str': self.diagnoses_str
        }
        prompt_grade_chat_msg = self.create_chat_message(role="user", content=prompts.prompt_grade_hybrid.substitute(subs))
        messages_lst = [self.sys_role_chat_msg, prompt_grade_chat_msg]

        # We should ideally use reasoning here, but it isn't enabled for the OpenAI API yet
        return getLLMJSON(messages_lst, self.client, self.model_name, prompts.cxrGradeHybrid)
    
    ##############################
    ### METHOD 4: LLM AS JUDGE ###
    ##############################

    def judgeGrading(self, rpt, grade_algo, grade_llm, grade_manual=None):
        return_json = {}

        # If the algo and LLM grades are the same, return the grade and choice 3
        if grade_algo == grade_llm:
            return_json['grade_int'] = grade_algo
            return_json['choice_int'] = 3
            return_json['explanation_int'] = None
        # Otherwise, we use the LLM to judge which is better
        else:
            subs = {
                'rpt': rpt,
                'grade_algo': grade_algo,
                'grade_llm': grade_llm
            }
            prompt_judge_chat_msg = self.create_chat_message(role="user", content=prompts.prompt_judge_grading.substitute(subs))
            messages_lst = [self.sys_role_chat_msg, prompt_judge_chat_msg]
            # We should ideally use reasoning here, but it isn't enabled for the OpenAI API yet
            grade_json = getLLMJSON(messages_lst, self.client, self.model_name, prompts.cxrGradeJudge)
            return_json['grade_int'] = grade_json['grade']
            return_json['choice_int'] = grade_json['choice']
            return_json['explanation_int'] = grade_json['explanation']
        
        # If no manual grade is provided, skip the extended judge step
        if grade_manual is None:
            return_json['grade_ext'] = None
            return_json['choice_ext'] = None
            return_json['explanation_ext'] = None
            
            return return_json
        
        # If the algo and manual grades are the same, return the grade and choice 3
        if grade_manual == return_json['grade_int']:
            return_json['grade_ext'] = return_json['grade_int']
            return_json['choice_ext'] = 3
            return_json['explanation_ext'] = None
        # Otherwise, we use the LLM to judge which is better
        else:
            subs = {
                'rpt': rpt,
                'grade_manual': grade_manual,
                'grade_auto': return_json['grade_int']
            }
            prompt_judge_chat_msg_ext = self.create_chat_message(role="user", content=prompts.prompt_judge_grading_ext.substitute(subs))
            messages_lst = [self.sys_role_chat_msg, prompt_judge_chat_msg_ext]
            # We should ideally use reasoning here, but it isn't enabled for the OpenAI API yet
            grade_json_ext = getLLMJSON(messages_lst, self.client, self.model_name, prompts.cxrGradeJudge)
            return_json['grade_ext'] = grade_json_ext['grade']
            return_json['choice_ext'] = grade_json_ext['choice']
            return_json['explanation_ext'] = grade_json_ext['explanation']

        return return_json

    # Apply judgeGrading function to each row and extract results
    def gradeReportJudge(self, rpt, grade_algo, grade_llm, grade_manual=None):
        result = self.judgeGrading(rpt, grade_algo, grade_llm, grade_manual)
        
        return_result = {
            'judge_grade': result['grade_int'],
            'judge_choice': result['choice_int'],
            'judge_reasoning': result['explanation_int'],
            'judge_grade_ext': result['grade_ext'],
            'judge_choice_ext': result['choice_ext'],
            'judge_reasoning_ext': result['explanation_ext']
        }

        if self.log_level == 1:
            print(f'Report: {rpt}')
            gt_text = f'({grade_manual})' if grade_manual is not None else '(No GT)'
            print(f'LLM/Algo (GT) grading: {grade_llm}/{grade_algo} {gt_text}')
            print(f'Judge grading: {return_result["judge_grade"]}\n')
            print(f'Judge reasoning: {return_result["judge_reasoning"]}\n')
            # Print which grade the LLM chose (manual or auto) based on choice_ext
            if return_result['judge_choice_ext'] == 0:
                print("LLM chose: Neither grade is appropriate")
            elif return_result['judge_choice_ext'] == 1:
                print("LLM chose: Algorithm grade is better")
            elif return_result['judge_choice_ext'] == 2:
                print("LLM chose: Manual grade is better")
            elif return_result['judge_choice_ext'] == 3:
                print("LLM chose: Both grades are the same")
            print('---')
        
        # Always return the Series, regardless of log_level
        return return_result
    
    ######## END CLASS ########
    
    #############################################
    ### SUPPLEMENT: LUNIT FINDINGS EXTRACTION ###
    #############################################

    def gradeLunit(self, rpt):
        """
        Extract Lunit-specific findings from a chest X-ray report.
        This supplemental method is designed to extract findings in a format
        compatible with Lunit AI analysis systems.
        
        Args:
            rpt (str): The chest X-ray report text to analyze.
        
        Returns:
            dict: A JSON object containing the Lunit findings extraction results.
        """
        subs = {
            'rpt': rpt
        }
        prompt_grade_chat_msg = self.create_chat_message(
            role="user", 
            content=prompts.prompt_get_lunit_findings.substitute(subs)
        )
        messages_lst = [self.sys_role_chat_msg, prompt_grade_chat_msg]

        return getLLMJSON(messages_lst, self.client, self.model_name, prompts.LunitFindings)
