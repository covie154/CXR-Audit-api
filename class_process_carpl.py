#%% 
import pandas as pd
import numpy as np
import os
import re
import json
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt


# Change to the scripts_lunit_review directory
#os.chdir('scripts_lunit_review')
#print(f"New working directory: {os.getcwd()}")

from open_protected_xlsx import open_protected_xlsx
from cxr_audit.lib_audit_cxr_v2 import CXRClassifier
from cxr_audit.grade_batch_async import BatchCXRProcessor

# Load semialgo data
with open("padchest_op.json", "r") as f:
    padchest = json.load(f)
    
with open("padchest_tubes_lines.json", "r") as f:
    tubes_lines = json.load(f)

with open("diagnoses.json", "r") as f:
    diagnoses = json.load(f)
    
# Initialize batch processor
processor = BatchCXRProcessor(
    findings_dict=padchest,
    tubes_lines_dict=tubes_lines,
    diagnoses_dict=diagnoses,
    model_name="qwen3:32b-q4_K_M",
    base_url="http://192.168.1.34:11434/v1",
    api_key="dummy",
    max_workers=4,  # Adjust based on your system and API limits
    rate_limit_delay=0  # Adjust based on your API rate limits
)

def calculate_agreement_metrics(ground_truth, predictions):
    """
    Calculate agreement metrics between ground truth and predictions.
    
    Args:
        ground_truth (array-like): Ground truth binary labels
        predictions (array-like): Predicted binary labels
        
    Returns:
        dict: Dictionary containing various agreement metrics
    """
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    
    # Calculate metrics
    exact_agreement = accuracy_score(ground_truth, predictions)
    cohen_kappa = cohen_kappa_score(ground_truth, predictions)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        'exact_agreement': exact_agreement,
        'cohen_kappa': cohen_kappa,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'positive_predictive_value': ppv,
        'negative_predictive_value': npv
    }


def perform_mcnemar_test(ground_truth, predictions1, predictions2):
    """
    Perform McNemar's test to compare two prediction methods.
    
    Args:
        ground_truth (array-like): Ground truth binary labels
        predictions1 (array-like): First method's predictions
        predictions2 (array-like): Second method's predictions
        
    Returns:
        dict: Dictionary containing McNemar test results
    """
    # Create binary correctness indicators
    correct1 = (predictions1 == ground_truth).astype(int)
    correct2 = (predictions2 == ground_truth).astype(int)
    
    # Create contingency table
    contingency_table = pd.crosstab(correct1, correct2, 
                                   margins=False, 
                                   rownames=['Method1'], 
                                   colnames=['Method2'])
    
    # Perform McNemar's test
    mcnemar_result = mcnemar(contingency_table, exact=True)
    
    # Calculate accuracy difference
    accuracy1 = correct1.mean()
    accuracy2 = correct2.mean()
    accuracy_diff = accuracy2 - accuracy1
    
    return {
        'contingency_table': contingency_table,
        'test_statistic': mcnemar_result.statistic,
        'p_value': mcnemar_result.pvalue,
        'significant': mcnemar_result.pvalue < 0.05,
        'accuracy_method1': accuracy1,
        'accuracy_method2': accuracy2,
        'accuracy_difference': accuracy_diff
    }
    
def convert_to_minutes(td):
    if pd.isnull(td):
        return np.nan
    time_mins = td.total_seconds() / 60
    time_secs = td.total_seconds() % 60
    return f"{int(time_mins)}m {int(time_secs)}s"

class ProcessCarpl:
    def __init__(self, path_carpl_reports, path_ge_reports, processor=processor, priority_threshold=10, passwd="GE_2024_P@55"):
        # Support both single file paths and lists of file paths
        self.path_carpl_reports = path_carpl_reports if isinstance(path_carpl_reports, list) else [path_carpl_reports]
        self.path_ge_reports = path_ge_reports if isinstance(path_ge_reports, list) else [path_ge_reports]
        self.passwd = passwd
        self.processor = processor
        self.priority_threshold = priority_threshold

    def load_reports(self):
        # Load multiple CARPL files and combine them
        carpl_dataframes = []
        for carpl_file in self.path_carpl_reports:
            print(f"Loading CARPL file: {carpl_file}")
            if carpl_file.endswith('.csv'):
                df_carpl = pd.read_csv(carpl_file)
            else:
                # Handle Excel files for CARPL as well
                df_carpl = pd.read_excel(carpl_file)
            carpl_dataframes.append(df_carpl)
            print(f"  -> Loaded {len(df_carpl)} records from {os.path.basename(carpl_file)}")
        
        # Combine all CARPL dataframes
        df_lunit = pd.concat(carpl_dataframes, ignore_index=True)
        print(f"Combined CARPL files: {len(df_lunit)} total records")
        
        # Load multiple GE reports and combine them
        ge_dataframes = []
        for ge_file in self.path_ge_reports:
            print(f"Loading GE file: {ge_file}")
            if ge_file.endswith('.csv'):
                df_ge = pd.read_csv(ge_file)
            elif ge_file.endswith(('.xlsx', '.xls')):
                # Use the protected Excel reader for GE files
                df_ge = open_protected_xlsx(ge_file, self.passwd)
            else:
                raise ValueError(f"Unsupported file format for GE file: {ge_file}")
            ge_dataframes.append(df_ge)
            print(f"  -> Loaded {len(df_ge)} records from {os.path.basename(ge_file)}")
        
        # Combine all GE dataframes
        df_reports = pd.concat(ge_dataframes, ignore_index=True)
        print(f"Combined GE files: {len(df_reports)} total records")
        
        print(f"Total loaded: {len(df_lunit)} CARPL records and {len(df_reports)} GE records. ", end="")
              
        df_merged = pd.merge(df_reports, df_lunit, left_on="ACCESSION_NO", right_on="Accession Number", how="inner")
        # Drop specified columns from df_merged
        # There are basically two parts to this exercise:
        # 1. Accuracy audit
        # 2. Time audit
        columns_to_drop = ['MEDICAL_LOCATION_CODE', 'PROCEDURE_NAME']
        df_merged = df_merged.drop(columns=columns_to_drop, errors='ignore')
        # Rename PROCEDURE_DATE to PROCEDURE_END_DATE if it exists
        if 'PROCEDURE_DATE' in df_merged.columns:
            df_merged = df_merged.rename(columns={'PROCEDURE_DATE': 'PROCEDURE_END_DATE'})
        
        print(f"Merged to {len(df_merged)} entries.")
        
        return df_merged
    
    def txt_initial_metrics(self, df_merged):
        # Determine the first and last date
        first_date = df_merged['PROCEDURE_START_DATE'].min()
        first_date_formatted = first_date.strftime("%d-%b-%Y")
        last_date = df_merged['PROCEDURE_START_DATE'].max()
        last_date_formatted = last_date.strftime("%d-%b-%Y")
        # Calculate the number of Non-CHESTs inferenced
        non_chest_count = len(df_merged[df_merged['PROCEDURE_CODE'] != 556])
        # Determine the total number of sites inferenced from
        sites_count = df_merged.groupby('MEDICAL_LOCATION_NAME').size()
        
        return f'''# REVIEW FOR PERIOD: {first_date_formatted} to {last_date_formatted}
Number of studies inferenced: {len(df_merged)}

# BASIC DESCRIPTORS
Number of inappropriate studies inferenced: {non_chest_count}
(Inappropriate studies include: Chest (Screening), AP/Oblique, AP/Lateral)

Studies were inferenced from the following sites:
{chr(10).join([f" - {site}: {count}" for site, count in sites_count.items()])}
        '''
        
    def process_stats_row(self, row, priority_threshold):
        # Process a single row to determine binary predictions based on priority threshold
        findings_columns = ["Atelectasis", "Calcification", "Cardiomegaly", \
                            "Consolidation", "Fibrosis", "Mediastinal Widening", \
                            "Nodule", "Pleural Effusion", "Pneumoperitoneum", \
                            "Pneumothorax", "Tuberculosis"]
        
        for column in findings_columns:
            new_column_name = column + "_binary"
            row[new_column_name] = 1 if row[column] > priority_threshold else 0

        # Create Overall_binary column
        binary_columns = [col + "_binary" for col in findings_columns]
        overall_binary = max([row[col] for col in binary_columns])

        return overall_binary

    def process_stats_accuracy(self, df_merged):
        df_merged['Overall_binary'] = df_merged.apply(lambda row: self.process_stats_row(row, self.priority_threshold), axis=1)

        # Compute LLM score and binarize it
        processed_reports = self.processor.process_full_pipeline(df_merged, report_column='TEXT_REPORT', steps=['llm'])
        processed_reports['llm_grade_binary'] = processed_reports['llm_grade'].apply(lambda x: 1 if x > 2 else 0)
        
        # merge the column 'llm_grade_binary' back to df_merged
        df_merged = pd.merge(df_merged, processed_reports[['ACCESSION_NO', 'llm_grade_binary']], on='ACCESSION_NO', how='left')
        
        return df_merged

    def txt_stats_accuracy(self, df_merged):
        if 'ground truth' not in df_merged.columns:
            print("GT not available. Testing LLM vs Lunit only.")
            df_reports_accuracy = df_merged[['ACCESSION_NO', 'Overall_binary', 'llm_grade_binary']]
            df_reports_accuracy.columns = ['accession', 'lunit', 'llm']
        else:           
            df_reports_accuracy = df_merged[['ACCESSION_NO', 'ground truth', 'Overall_binary', 'llm_grade_binary']]
            df_reports_accuracy.columns = ['accession', 'gt', 'lunit', 'llm']
            gt = df_reports_accuracy['gt']

        lunit = df_reports_accuracy['lunit']
        llm = df_reports_accuracy['llm']
        output_txt = ""

        # If GT is present, 
        # Calculate agreement metrics for GT vs Lunit
        if 'gt' in df_reports_accuracy.columns:
            gt_lunit_metrics = calculate_agreement_metrics(gt, lunit)
            
            output_txt += f'''=== GROUND TRUTH vs LUNIT ANALYSIS (ORIGINAL METHOD) ===
Exact Agreement (GT vs Lunit): {gt_lunit_metrics['exact_agreement']:.3f} ({gt_lunit_metrics['exact_agreement']*100:.1f}%)
Cohen's Kappa (GT vs Lunit): {gt_lunit_metrics['cohen_kappa']:.3f}
Sensitivity (GT vs Lunit): {gt_lunit_metrics['sensitivity']:.3f}
Specificity (GT vs Lunit): {gt_lunit_metrics['specificity']:.3f}
PPV (GT vs Lunit): {gt_lunit_metrics['positive_predictive_value']:.3f}
NPV (GT vs Lunit): {gt_lunit_metrics['negative_predictive_value']:.3f}

'''
        
        # Calculate agreement metrics for LLM vs Lunit
        llm_lunit_metrics = calculate_agreement_metrics(llm, lunit)
        
        output_txt += f'''=== LLM vs LUNIT ANALYSIS (NEW METHOD) ===
Exact Agreement (LLM vs Lunit): {llm_lunit_metrics['exact_agreement']:.3f} ({llm_lunit_metrics['exact_agreement']*100:.1f}%)
Cohen's Kappa (LLM vs Lunit): {llm_lunit_metrics['cohen_kappa']:.3f}
Sensitivity (LLM vs Lunit): {llm_lunit_metrics['sensitivity']:.3f}
Specificity (LLM vs Lunit): {llm_lunit_metrics['specificity']:.3f}
PPV (LLM vs Lunit): {llm_lunit_metrics['positive_predictive_value']:.3f}
NPV (LLM vs Lunit): {llm_lunit_metrics['negative_predictive_value']:.3f}

'''

        # If GT is present,
        # Calculate agreement metrics for GT vs LLM
        if 'gt' in df_reports_accuracy.columns:
            gt_llm_metrics = calculate_agreement_metrics(gt, llm)
            # Perform McNemar test between gt and LLM
            mcnemar_result = perform_mcnemar_test(lunit, gt, llm)
            
            output_txt += f'''=== GROUND TRUTH vs LLM ANALYSIS (VALIDATION)===
Exact Agreement (GT vs LLM): {gt_llm_metrics['exact_agreement']:.3f} ({gt_llm_metrics['exact_agreement']*100:.1f}%)
Cohen's Kappa (GT vs LLM): {gt_llm_metrics['cohen_kappa']:.3f}

=== McNEMAR'S TEST (GT vs LLM) ===
Contingency Table (GT rows, LLM columns):
0 = Incorrect, 1 = Correct
{mcnemar_result['contingency_table']}
Test Statistic: {mcnemar_result['test_statistic']:.3f}
P-value: {mcnemar_result['p_value']:.6f}
Statistically Significant: {'Yes' if mcnemar_result['significant'] else 'No'}
Accuracy Difference (LLM - GT): {mcnemar_result['accuracy_difference']:.3f}
'''

        # Print results
        return output_txt
        
    def process_stats_time(self, df_merged):
        # For Time to Clinical Decision
        # If the flag is not "5 AI_ROUTINE", take AI_FLAG_RECEIVED_DATE - PROCEDURE_END_DATE,
        # Otherwise just copy TAT

        # For End to End Server Time
        # If the flag is not "4 URGENT", take AI_FLAG_RECEIVED_DATE - PROCEDURE_END_DATE,
        # Otherwise 0

        def time_clinical_decision(row):
            if row['AI_PRIORITY'] == "5 AI_ROUTINE":
                return row['AI_FLAG_RECEIVED_DATE'] - row['PROCEDURE_END_DATE']
            else:
                return row['REPORT_TURN_AROUND_TIME']

        def time_end_to_end(row):
            if row['AI_PRIORITY'] != "4 URGENT":
                return row['AI_FLAG_RECEIVED_DATE'] - row['PROCEDURE_END_DATE']
            else:
                return 0
            
        df_merged['Time_to_Clinical_Decision'] = df_merged.apply(time_clinical_decision, axis=1)
        df_merged['Time_End_to_End'] = df_merged.apply(time_end_to_end, axis=1)
        # Convert float minutes to pandas.Timedelta
        df_merged['Time_to_Clinical_Decision'] = pd.to_timedelta(df_merged['Time_to_Clinical_Decision'], unit='seconds')
        df_merged['Time_End_to_End'] = pd.to_timedelta(df_merged['Time_End_to_End'], unit='seconds')
        
        return df_merged
    
    def txt_stats_time(self, df_merged):
        time_end_to_end_nonzero = df_merged[df_merged['Time_End_to_End'] != pd.Timedelta(0)]['Time_End_to_End'].dropna()
        cases_late = len(time_end_to_end_nonzero[time_end_to_end_nonzero > pd.Timedelta(minutes=5)])
        total_len = len(df_merged)
        
        # Print summary statistics
        return f'''
=== TIME ANALYSIS SUMMARY ===
Time to Clinical Decision:
(time from Exam End to AI Flag Received (i.e. case processed), otherwise Report TAT (i.e. case not processed))
- Mean: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].mean())}
- Median: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].median())}
- Std: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].std())}
- Min: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].min())}
- Max: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].max())}

End to End Server Time:
(time from Exam End to AI Flag Received, excluding cases where the flag never made it)
- Mean: {convert_to_minutes(time_end_to_end_nonzero.mean())}
- Median: {convert_to_minutes(time_end_to_end_nonzero.median())}
- Std: {convert_to_minutes(time_end_to_end_nonzero.std())}
- Min: {convert_to_minutes(time_end_to_end_nonzero.min())}
- Max: {convert_to_minutes(time_end_to_end_nonzero.max())}
- % of cases with t > 5 mins: {cases_late / total_len * 100:.2f}% ({cases_late} / {total_len} case(s))
        '''
        
    def box_time(self, df_merged):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Box plot for Time_to_Clinical_Decision
        axes[0].boxplot(df_merged['Time_to_Clinical_Decision'].dropna())
        axes[0].set_title('Time to Clinical Decision')
        axes[0].set_ylabel('Time (minutes)')
        axes[0].grid(True, alpha=0.3)

        # Box plot for Time_End_to_End (excluding zero values)
        time_end_to_end_nonzero = df_merged[df_merged['Time_End_to_End'] != pd.Timedelta(0)]['Time_End_to_End'].dropna()
        axes[1].boxplot(time_end_to_end_nonzero)
        axes[1].set_title('End to End Server Time')
        axes[1].set_ylabel('Time (minutes)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def calculate_tn_fp_fn_tp(self, row, gt_col='ground truth', priority_threshold=10):
        overall_binary = self.process_stats_row(row, priority_threshold)

        gt = row[gt_col]
        pred = overall_binary
        
        if gt == 1 and pred == 1:
            return pd.Series({f'Lunit {priority_threshold}': pred, 'TP': 1, 'TN': 0, 'FP': 0, 'FN': 0})
        elif gt == 0 and pred == 0:
            return pd.Series({f'Lunit {priority_threshold}': pred, 'TP': 0, 'TN': 1, 'FP': 0, 'FN': 0})
        elif gt == 0 and pred == 1:
            return pd.Series({f'Lunit {priority_threshold}': pred, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 1})
        elif gt == 1 and pred == 0:
            return pd.Series({f'Lunit {priority_threshold}': pred, 'TP': 0, 'TN': 0, 'FP': 1, 'FN': 0})
        else:
            return pd.Series({f'Lunit {priority_threshold}': pred, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0})
        
    def highest_probability(self, row):
        findings_columns = ["Atelectasis", "Calcification", "Cardiomegaly", \
                            "Consolidation", "Fibrosis", "Mediastinal Widening", \
                            "Nodule", "Pleural Effusion", "Pneumoperitoneum", \
                            "Pneumothorax", "Tuberculosis"]
        
        probabilities = [row[column] for column in findings_columns]
        max_prob = max(probabilities)
        max_index = probabilities.index(max_prob)
        return findings_columns[max_index]

    def rearrange_columns(self, df):
        '''Column order should be:
        [study_id, PatientID, PatientName, ai_status, gt_status, feedback, comments, AccessionNumber, 
        Abnormal, Atelectasis, Calcification, Cardiomegaly, Consolidation, Fibrosis, Mediastinal Widening, Nodule, Pleural Effusion, Pneumoperitoneum, Pneumothorax, Tuberculosis, 
        AI Report, Lunit 10, ground truth, TP 10, TN 10, FP10, FN 10, <blank>, 
        Location, Date, Age, CXR report, highest probability, <blank>, 
        Lunit 7.5, TP 7.5, TN 7.5, FP 7.5, FN 7.5, Lunit 12.5, TP 12.5, TN 12.5, FP 12.5, FN 12.5, Lunit 15, TP15, TN 15, FP 15, FN 15, 
        PROCEDURE_START_DATE, PROCEDURE_END_DATE, AI_PRIORITY, AI_FLAG_RECEIVED_DATE, TAT, Time_to_clinical_decision(mins), End to End server TAT, % End to End < 5 mins]
        
        Lunit 10 = Overall_binary
        '''
        
        # First we add the missing columns: 
        # TP 10, TN 10, FP10, FN 10, 
        # Lunit 7.5, TP 7.5, TN 7.5, FP 7.5, FN 7.5, 
        # Lunit 12.5, TP 12.5, TN 12.5, FP 12.5, FN 12.5, 
        # Lunit 15, TP15, TN 15, FP 15, FN 15
        
        thresholds = [7.5, 10, 12.5, 15]
        for threshold in thresholds:
            df[[f'Lunit {threshold}', f'TP {threshold}', f'TN {threshold}', f'FP {threshold}', f'FN {threshold}']] = \
                df.apply(lambda row: self.calculate_tn_fp_fn_tp(row, priority_threshold=threshold, gt_col='llm_grade_binary'), axis=1)
        
        # If 'ground truth' is missing, we add it as NaN
        if 'ground truth' not in df.columns:
            df['ground truth'] = np.nan
        # Add the column "ai_status", fill with "Processed"
        df['ai_status'] = 'Processed'
        # Add the column "gt_status", fill with "LLM Processed"
        df['gt_status'] = 'LLM Processed'
        # Duplicate the column "PROCEDURE_END_DATE" to "date"
        df['date'] = df['PROCEDURE_END_DATE']
        # Add a spacer column
        df['spacer'] = ''
        # Calculate the column "highest_probability"
        df['highest_probability'] = df.apply(self.highest_probability, axis=1)

        # NB: We swap the column "llm_grade_binary" into the GT column, and leave the original GT column behind "highest_probability" as "ground_truth"
        # PATIENT_GENDER is missing from the original csv, so we add it as the last column
        new_column_order = ['StudyID', 'Patient ID', 'Patient Name', 'ai_status', 'gt_status', 'Feedback', 'Comments', 'ACCESSION_NO', 
                            'Abnormal', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'Fibrosis', 'Mediastinal Widening', 
                            'Nodule', 'Pleural Effusion', 'Pneumoperitoneum', 'Pneumothorax', 'Tuberculosis', 
                            'AI Report', 'Lunit 10', 'llm_grade_binary', 'TP 10', 'TN 10', 'FP 10', 'FN 10', 'spacer',
                            'MEDICAL_LOCATION_NAME', 'date', 'PATIENT_AGE', 'TEXT_REPORT', 'highest_probability', 'ground truth', 
                            'Lunit 7.5', 'TP 7.5', 'TN 7.5', 'FP 7.5', 'FN 7.5', 
                            'Lunit 12.5', 'TP 12.5', 'TN 12.5', 'FP 12.5', 'FN 12.5', 
                            'Lunit 15', 'TP 15', 'TN 15', 'FP 15', 'FN 15',
                            'PROCEDURE_START_DATE', 'PROCEDURE_END_DATE', 
                            'AI_PRIORITY', 'AI_FLAG_RECEIVED_DATE', 
                            'REPORT_TURN_AROUND_TIME', 'Time_to_Clinical_Decision',
                            'Time_End_to_End', 'PATIENT_GENDER']

        df = df[new_column_order]
        
        # Now we rename the columns to fit the original format
        df.columns = ['study_id', 'PatientID', 'PatientName', 'ai_status', 'gt_status', 'feedback', 'comments', 'AccessionNumber', 
                      'Abnormal', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'Fibrosis', 'Mediastinal Widening', 
                      'Nodule', 'Pleural Effusion', 'Pneumoperitoneum', 'Pneumothorax', 'Tuberculosis', 
                      'AI Report', 'Lunit 10', 'ground truth', 'TP 10', 'TN 10', 'FP 10', 'FN 10', 'spacer',
                      'Location', 'Date', 'Age', 'CXR report', 'highest probability', 'ground truth original',
                      'Lunit 7.5', 'TP 7.5', 'TN 7.5', 'FP 7.5', 'FN 7.5',
                      'Lunit 12.5', 'TP 12.5', 'TN 12.5', 'FP 12.5', 'FN 12.5',
                      'Lunit 15', 'TP 15', 'TN 15', 'FP 15', 'FN 15',
                      'PROCEDURE_START_DATE', 'PROCEDURE_END_DATE',
                      'AI_PRIORITY', 'AI_FLAG_RECEIVED_DATE',
                      'TAT', 'Time_to_clinical_decision(mins)',
                      'End to End server TAT', 'Gender']

        return df

    def identify_false_negatives(self, df_merged):
        """
        Identify false negative reports where LLM predicts negative (0) but Lunit predicts positive (1).
        
        Args:
            df_merged: DataFrame containing both llm_grade_binary and Overall_binary columns
            
        Returns:
            tuple: (false_negative_df, false_negative_summary)
        """
        # Ensure we have the required columns
        if 'llm_grade_binary' not in df_merged.columns or 'Overall_binary' not in df_merged.columns:
            return pd.DataFrame(), {"count": 0, "percentage": 0}
        
        # Identify false negatives: LLM=0 (negative) and Lunit=1 (positive)
        false_negatives = df_merged[
            (df_merged['llm_grade_binary'] == 0) & 
            (df_merged['Overall_binary'] == 1)
        ]
        
        # Select relevant columns for the false negative report
        fn_columns = [
            'ACCESSION_NO', 'PATIENT_AGE', 'MEDICAL_LOCATION_NAME', 
            'PROCEDURE_START_DATE', 'TEXT_REPORT', 'llm_grade_binary', 'Overall_binary',
            'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 
            'Fibrosis', 'Mediastinal Widening', 'Nodule', 'Pleural Effusion', 
            'Pneumoperitoneum', 'Pneumothorax', 'Tuberculosis'
        ]
        
        # Filter to only include columns that exist in the dataframe
        available_columns = [col for col in fn_columns if col in df_merged.columns]
        false_negative_df = false_negatives[available_columns].copy()
        
        # Add highest probability finding
        if not false_negative_df.empty:
            false_negative_df['highest_probability'] = false_negative_df.apply(self.highest_probability, axis=1)
        
        # Create summary
        total_cases = len(df_merged)
        fn_count = len(false_negative_df)
        fn_percentage = (fn_count / total_cases * 100) if total_cases > 0 else 0
        
        false_negative_summary = {
            "count": fn_count,
            "total_cases": total_cases,
            "percentage": fn_percentage,
            "description": f"Cases where LLM predicted negative (0) but Lunit predicted positive (1)"
        }
        
        return false_negative_df, false_negative_summary

    def run_all(self):
        df_merged = self.load_reports()
        print(self.txt_initial_metrics(df_merged))
        
        df_merged = self.process_stats_accuracy(df_merged)
        print(self.txt_stats_accuracy(df_merged))

        df_merged = self.process_stats_time(df_merged)
        print(self.txt_stats_time(df_merged))
        
        self.box_time(df_merged)
        
        # Identify false negatives before rearranging columns
        false_negative_df, false_negative_summary = self.identify_false_negatives(df_merged)
        
        # Finally we rearrange the columns to fit the proper format
        df_merged = self.rearrange_columns(df_merged)
        
        return df_merged, false_negative_df, false_negative_summary
    
'''
# Get the current working directory and navigate to data_audit folder
current_dir = os.getcwd()
base_path = os.path.abspath(os.path.join(current_dir, '..', 'data_lunit_review'))
data_file = 'deployment_stats14.csv'
path_carpl = os.path.join(base_path, data_file)

path_reports = "../data_lunit_review/07-Aug-2025/07-Aug-2025"
reports_file = "RIS_WeeklyReport_07Aug2025.xls"
path_reports = os.path.join(path_reports, reports_file)

process_carpl = ProcessCarpl(
    path_carpl_reports=path_carpl,
    path_ge_reports=path_reports,
    processor=processor,
)

df_merged = process_carpl.run_all()
'''

#%%