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
    base_url="http://192.168.1.204:11434/v1",
    api_key="dummy",
    max_workers=8,  # Adjust based on your system and API limits
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
    roc_auc = (sensitivity + specificity) / 2

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
        'negative_predictive_value': npv,
        'roc_auc': roc_auc
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

thresholds_default = {
    'default': {
        'Atelectasis': 10,
        'Calcification': 10,
        'Cardiomegaly': 10,
        'Consolidation': 10,
        'Fibrosis': 10,
        'Mediastinal Widening': 10,
        'Nodule': 10,
        'Pleural Effusion': 10,
        'Pneumoperitoneum': 10,
        'Pneumothorax': 10,
        'Tuberculosis': 999
    }
}

class ProcessCarpl:
    def __init__(self, path_carpl_reports, path_ge_reports, processor=processor, supplemental_steps=True, priority_threshold=thresholds_default, passwd="GE_2024_P@55"):
        # Support both single file paths and lists of file paths
        self.path_carpl_reports = path_carpl_reports if isinstance(path_carpl_reports, list) else [path_carpl_reports]
        self.path_ge_reports = path_ge_reports if isinstance(path_ge_reports, list) else [path_ge_reports]
        self.passwd = passwd
        self.processor = processor
        self.priority_threshold = priority_threshold
        self.supplemental_steps = supplemental_steps

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
                try:
                    df_ge = open_protected_xlsx(ge_file, self.passwd)
                except Exception as e:
                    try:
                        print(f"{ge_file} doesn't have a password, trying without password: {e}")
                        df_ge = pd.read_excel(ge_file)
                    except Exception as e:
                        print(f"Error loading protected Excel file {ge_file}: {e}")
                        continue
            else:
                raise ValueError(f"Unsupported file format for GE file: {ge_file}")
            
            # Rename PROCEDURE_DATE to PROCEDURE_END_DATE if it exists
            if 'PROCEDURE_DATE' in df_ge.columns:
                df_ge = df_ge.rename(columns={'PROCEDURE_DATE': 'PROCEDURE_END_DATE'})

            # Then append to list
            ge_dataframes.append(df_ge)
            print(f"  -> Loaded {len(df_ge)} records from {os.path.basename(ge_file)}")
        
        # Combine all GE dataframes
        df_reports = pd.concat(ge_dataframes, ignore_index=True)
        print(f"Combined GE files: {len(df_reports)} total records")
        
        # Remove duplicate ACCESSION_NO entries, keeping the first occurrence
        duplicates_count = df_reports['ACCESSION_NO'].duplicated().sum()
        if duplicates_count > 0:
            print(f"  -> Found {duplicates_count} duplicate ACCESSION_NO entries in GE files")
            df_reports = df_reports[df_reports['ACCESSION_NO'].duplicated() == False]
            print(f"  -> Removed duplicates, now {len(df_reports)} unique GE records")
        
        print(f"Total loaded: {len(df_lunit)} CARPL records and {len(df_reports)} GE records. ", end="")
        

        # Convert AI_FLAG_RECEIVED_DATE to datetime
        df_reports['AI_FLAG_RECEIVED_DATE'] = pd.to_datetime(df_reports['AI_FLAG_RECEIVED_DATE'], errors='coerce')
        
        # ADD: Convert PROCEDURE_START_DATE and PROCEDURE_END_DATE to datetime
        df_reports['PROCEDURE_START_DATE'] = pd.to_datetime(df_reports['PROCEDURE_START_DATE'], errors='coerce')
        df_reports['PROCEDURE_END_DATE'] = pd.to_datetime(df_reports['PROCEDURE_END_DATE'], errors='coerce')
        
        # Convert all cells with "-" to np.nan
        df_lunit = df_lunit.replace("-", np.nan)
        # Apply proper dtypes
        df_lunit = self._apply_column_dtypes(df_lunit)
        
        # IMPORTANT: Ensure ACCESSION_NO types match before merge to avoid type mismatch issues
        # Convert both to string first, then to Int64 for consistent matching
        df_reports['ACCESSION_NO'] = pd.to_numeric(df_reports['ACCESSION_NO'], errors='coerce').astype('Int64')
        df_lunit['Accession Number'] = pd.to_numeric(df_lunit['Accession Number'], errors='coerce').astype('Int64')
        
        # Debug: Show accession number types and sample values
        print(f"  -> GE ACCESSION_NO dtype: {df_reports['ACCESSION_NO'].dtype}, sample: {df_reports['ACCESSION_NO'].head(3).tolist()}")
        print(f"  -> CARPL Accession Number dtype: {df_lunit['Accession Number'].dtype}, sample: {df_lunit['Accession Number'].head(3).tolist()}")
        
        df_merged = pd.merge(df_reports, df_lunit, left_on="ACCESSION_NO", right_on="Accession Number", how="inner")
        # Remove duplicate accession numbers, keeping the first occurrence
        df_merged = df_merged.drop_duplicates(subset='ACCESSION_NO', keep='first')
        print(f"Merged data has {len(df_merged)} records after removing duplicates.")
        
        # Drop specified columns from df_merged
        # There are basically two parts to this exercise:
        # 1. Accuracy audit
        # 2. Time audit
        columns_to_drop = ['MEDICAL_LOCATION_CODE', 'PROCEDURE_NAME']
        df_merged = df_merged.drop(columns=columns_to_drop, errors='ignore')
        
        # Merge PROCEDURE_DATE into PROCEDURE_END_DATE if both exist
        if 'PROCEDURE_DATE' in df_merged.columns and 'PROCEDURE_END_DATE' in df_merged.columns:
            df_merged['PROCEDURE_END_DATE'] = df_merged['PROCEDURE_END_DATE'].fillna(df_merged['PROCEDURE_DATE'])
            df_merged = df_merged.drop(columns=['PROCEDURE_DATE'])
        
        print(f"Merged to {len(df_merged)} entries.")
        
        return df_merged
    
    def _apply_column_dtypes(self, df):
        """
        Apply proper data types to columns for consistency and memory efficiency.
        
        Args:
            df: DataFrame to apply dtypes to
            
        Returns:
            DataFrame with corrected dtypes
        """
        dtype_mapping = {
            # Identifiers - int64
            'ACCESSION_NO': 'Int64',  # Nullable integer to handle NaN
            'Accession Number': 'Int64',
            
            # Patient info - strings
            'Patient Name': 'string',
            'PATIENT_GENDER': 'category',
            'WORKPLACE': 'category',
            'MEDICAL_LOCATION_NAME': 'category',
            
            # Age - int
            'PATIENT_AGE': 'Int64',
            
            # Reports - strings
            'TEXT_REPORT': 'string',
            'AI Report': 'string',
            'Comments': 'string',
            'Feedback': 'string',
            
            # Dates - datetime
            'PROCEDURE_START_DATE': 'datetime64[ns]',
            'PROCEDURE_END_DATE': 'datetime64[ns]',
            'AI_FLAG_RECEIVED_DATE': 'datetime64[ns]',
            
            # Time intervals - timedelta
            'REPORT_TURN_AROUND_TIME': 'timedelta64[ns]',
            
            # Binary/categorical fields
            'AI_PRIORITY': 'category',
            'PROCEDURE_CODE': 'Int64',
            'ai_status': 'category',
            'gt_status': 'category',
            
            # Lunit probability scores - float
            'Atelectasis': 'float32',
            'Calcification': 'float32',
            'Cardiomegaly': 'float32',
            'Consolidation': 'float32',
            'Fibrosis': 'float32',
            'Mediastinal Widening': 'float32',
            'Nodule': 'float32',
            'Pleural Effusion': 'float32',
            'Pneumoperitoneum': 'float32',
            'Pneumothorax': 'float32',
            'Tuberculosis': 'float32',
            'Abnormal': 'float32',
            
            # Binary predictions - int8
            'Overall_binary': 'Int8',
            'llm_grade_binary': 'Int8',
            'lunit_grade': 'Int8',
            'ground truth': 'Int8',
            
            # Highest probability
            'highest_probability': 'category',
        }
        
        # Apply dtypes only for columns that exist
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                try:
                    if dtype in ['datetime64[ns]']:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif dtype in ['timedelta64[ns]']:
                        # Keep as is if already timedelta
                        if not pd.api.types.is_timedelta64_dtype(df[col]):
                            df[col] = pd.to_timedelta(df[col], errors='coerce')
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert {col} to {dtype}: {e}")
        
        return df
    
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
        '''
        # UNUSED FOR NOW
        '''
Studies were inferenced from the following sites:
{chr(10).join([f" - {site}: {count}" for site, count in sites_count.items()])}
        '''
    
    def transform_workplace(self, workplace):
        mapping = {
            'GLCR01': 'GEY',
            'KALCR01': 'KAL',
            'SEMCR01': 'SEM',
            'TPYCR01': 'TPY',
            'KHACR01': 'KHA',
            'WDLCR01': 'WDL',
            'HOUCR01': 'HOU',
            'AMKCR01': 'AMK',
            'YISCR01': 'YIS'
        }
        
        return mapping.get(workplace, 'OTH')
    
    def fill_threshold_dict(self, threshold_dict_site):
        '''
        Fill the priority threshold dictionary with default values for missing findings
        For finding in lunit findings list, if not in threshold_dict_site, use default from self.thresholds_default
        '''
        
        # VALIDATION
        # If any finding in threshold_dict_site is not in self.thresholds_default, raise error
        findings_columns = ["Atelectasis", "Calcification", "Cardiomegaly", \
                            "Consolidation", "Fibrosis", "Mediastinal Widening", \
                            "Nodule", "Pleural Effusion", "Pneumoperitoneum", \
                            "Pneumothorax", "Tuberculosis"]
        
        for finding in threshold_dict_site.keys():
            if finding not in findings_columns:
                raise ValueError(f"Invalid finding in threshold_dict_site: {finding}")
        
        # If the threshold is not a number, raise error
        for finding, threshold in threshold_dict_site.items():
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"Invalid threshold value for {finding}: {threshold}")
        
        for finding, default_threshold in self.priority_threshold['default'].items():
            if finding not in threshold_dict_site:
                threshold_dict_site[finding] = default_threshold
        
        return threshold_dict_site
            
    def process_stats_row(self, row, priority_threshold_dict):
        # Process a single row to determine binary predictions based on priority threshold
        # Also depends on site-specific thresholds
        
        
        site = self.transform_workplace(row['WORKPLACE'])
        if site == 'OTH' or site not in priority_threshold_dict:
            threshold_dict_site = self.fill_threshold_dict(priority_threshold_dict['default'])
        else:
            threshold_dict_site = self.fill_threshold_dict(priority_threshold_dict[site])
        
        temp_scores = 0
        for finding, threshold in threshold_dict_site.items():
            temp_scores += 1 if row[finding] > threshold else 0
        
        return 1 if temp_scores > 0 else 0

    def process_stats_accuracy(self, df_merged, supplemental_steps=False):        
        # First, compute the overall binary for pred based on thresholds
        df_merged['Overall_binary'] = df_merged.apply(lambda row: self.process_stats_row(row, self.priority_threshold), axis=1)
        
        if supplemental_steps:
            steps = ['llm', 'lunit']
        else:
            steps = ['llm']

        # Compute LLM score and binarize it
        processed_reports = self.processor.process_full_pipeline(df_merged, report_column='TEXT_REPORT', steps=steps)
        processed_reports['llm_grade_binary'] = processed_reports['llm_grade'].apply(lambda x: 1 if x > 1 else 0)
        
        # merge the column 'llm_grade_binary' back to df_merged
        if supplemental_steps:
            columns_to_merge = ['ACCESSION_NO', 'llm_grade_binary', 'atelectasis_llm', 'calcification_llm', 
                                'cardiomegaly_llm', 'consolidation_llm', 'fibrosis_llm', 'mediastinal_widening_llm', 
                                'nodule_llm', 'pleural_effusion_llm', 'pneumoperitoneum_llm', 'pneumothorax_llm', 'tb']
        else:
            columns_to_merge = ['ACCESSION_NO', 'llm_grade_binary']
            
        df_merged = pd.merge(df_merged, processed_reports[columns_to_merge], on='ACCESSION_NO', how='left')
        return df_merged

    def txt_stats_accuracy(self, df_merged, gt_column='llm_grade_binary', pred_column='Overall_binary', manual_gt_column='ground truth'):
        """
        Generate accuracy statistics text report.
        
        Args:
            df_merged: DataFrame with processed reports
        """
        
        workplace_groups = df_merged.groupby('WORKPLACE')
        workplace_dfs = {workplace: group for workplace, group in workplace_groups}
        
        all_stats_df = pd.DataFrame()
        for workplace, df_workplace in workplace_dfs.items():
            stats = calculate_agreement_metrics(
                ground_truth=df_workplace[gt_column],
                predictions=df_workplace[pred_column]
            )
            
            stats_df = pd.DataFrame(stats, index=[self.transform_workplace(workplace)])
            stats_df = stats_df.rename(columns={
                'exact_agreement': 'accuracy',
                'true_positives': 'TP',
                'true_negatives': 'TN',
                'false_positives': 'FP',
                'false_negatives': 'FN',
                'positive_predictive_value': 'PPV',
                'negative_predictive_value': 'NPV',
                'roc_auc': 'ROC-AUC'
            })
            stats_df['n_cases'] = len(df_workplace)
            stats_df['%_normal'] = (df_workplace['Overall_binary'] == 0).mean() * 100
            stats_df = stats_df.round(3)
            all_stats_df = pd.concat([all_stats_df, stats_df])
            
        output_txt = '=== LLM vs LUNIT ANALYSIS (NEW METHOD, BY WORKPLACE) ===\n' + all_stats_df.to_string() + "\n\n"
        
        if manual_gt_column not in df_merged.columns:
            print("GT not available. Testing LLM vs Lunit only.")
            return output_txt
        
        # If manual GT is present, do the same for GT vs Lunit
        all_stats_df_gt = pd.DataFrame()
        for workplace, df_workplace in workplace_dfs.items():
            stats_gt = calculate_agreement_metrics(
                ground_truth=df_workplace['ground truth'],
                predictions=df_workplace['Overall_binary']
            )
            stats_gt_df = pd.DataFrame(stats_gt, index=[self.transform_workplace(workplace)])
            stats_gt_df = stats_gt_df.rename(columns={
                'exact_agreement': 'accuracy',
                'true_positives': 'TP',
                'true_negatives': 'TN',
                'false_positives': 'FP',
                'false_negatives': 'FN',
                'positive_predictive_value': 'PPV',
                'negative_predictive_value': 'NPV',
                'roc_auc': 'ROC-AUC'
            })
            stats_gt_df['n_cases'] = len(df_workplace)
            stats_gt_df['%_normal'] = (df_workplace['Overall_binary'] == 0).mean() * 100
            stats_gt_df = stats_gt_df.round(3)
            all_stats_df_gt = pd.concat([all_stats_df_gt, stats_gt_df])
        
        output_txt += '=== GROUND TRUTH vs LUNIT ANALYSIS (ORIGINAL METHOD, BY WORKPLACE) ===\n' + all_stats_df_gt.to_string() + "\n\n"
        
        # Also calculate the agreement between GT and LLM (Overall)
        gt_llm_metrics = calculate_agreement_metrics(
            ground_truth=df_merged['ground truth'],
            predictions=df_merged['llm_grade_binary']
        )
        mcnemar_result = perform_mcnemar_test(
            ground_truth=df_merged[gt_column],
            predictions1=df_merged[manual_gt_column],
            predictions2=df_merged[pred_column]
        )
        
        
        output_txt += f'''=== GROUND TRUTH vs LLM ANALYSIS (VALIDATION, OVERALL)===
Exact Agreement (GT vs LLM): {gt_llm_metrics['exact_agreement']:.3f} ({gt_llm_metrics['exact_agreement']*100:.1f}%)
Cohen's Kappa (GT vs LLM): {gt_llm_metrics['cohen_kappa']:.3f}

=== McNEMAR'S TEST (GT vs LLM, OVERALL) ===
Contingency Table (GT rows, LLM columns):
0 = Incorrect, 1 = Correct
{mcnemar_result['contingency_table']}
Test Statistic: {mcnemar_result['test_statistic']:.3f}
P-value: {mcnemar_result['p_value']:.6f}
Statistically Significant: {'Yes' if mcnemar_result['significant'] else 'No'}
Accuracy Difference (LLM - GT): {mcnemar_result['accuracy_difference']:.3f}
'''
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
        output_txt = f'''=== TIME ANALYSIS SUMMARY ===
Time to Clinical Decision:
(time from Exam End to AI Flag Received (i.e. case processed), otherwise Report TAT (i.e. case not processed))
- Mean: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].mean())}
- Median: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].median())}
- Std: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].std())}
- Min: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].min())}
- Max: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].max())}

End to End Server Time:
(time from Exam End to AI Flag Received, excluding cases where the flag never made it)
    '''

        if cases_late == 0:
            output_txt += f'- All {total_len} cases processed within 5 minutes'
        else:
            output_txt += f'''
- Mean: {convert_to_minutes(time_end_to_end_nonzero.mean())}
- Median: {convert_to_minutes(time_end_to_end_nonzero.median())}
- Std: {convert_to_minutes(time_end_to_end_nonzero.std())}
- Min: {convert_to_minutes(time_end_to_end_nonzero.min())}
- Max: {convert_to_minutes(time_end_to_end_nonzero.max())}
- % of cases with t > 5 mins: {cases_late / total_len * 100:.2f}% ({cases_late} / {total_len} case(s))
        '''
        return output_txt
        
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
        '''
        Original intended columns:
        [StudyIDAnonymized, StudyID, Patient Name, Patient ID, Instances, Age, Gender, Study Description, Upload Date, Accession Number,
        Abnormal, Atelectasis, Calcification, Cardiomegaly, Consolidation, Fibrosis, Mediastinal Widening, Nodule, Pleural Effusion, Pneumoperitoneum, Pneumothorax, Tuberculosis,
        Feedback, Comments, Status, Inference Date, AI Report, work station, PROCEDURE_START_DATE, PROCEDURE_END_DATE, AI_PRIORITY, AI_FLAG_RECEIVED_DATE, TAT, Time_to_clinical_decision(mins), 
        ground truth, Lunit0.1wTB, TP 0.1wTB, TN 0.1wTB, FP 0.1wTB, FN 0.1wTB, 
        ground truth, Lunit0.1woTB, TP 0.1woTB, TN 0.1woTB, FP 0.1woTB, FN 0.1woTB, 
        ground truth, Lunit0.15woTB, TP 0.15woTB, TN 0.15woTB, FP 0.15woTB, FN 0.15woTB, 
        ground truth, Lunit noTB NDL 0.15, TP noTB NDL0.15, TN noTB NDL0.15, FP noTB NDL0.15, FN noTB NDL0.15, 
        ground truth, Lunit noTB NDL 0.2, TP noTB NDL0.20, TN noTB NDL0.20, FP noTB NDL0.20, FN noTB NDL0.20, 
        ground truth , YIS nod .05, TP YISnod0.05, TN YISnod0.05, FP YISnod0.05, FN YISnod0.05, 
        check, REPORT]
        
        Column order should be:
        [StudyIDAnonymized, StudyID, Patient Name, Patient ID, Instances, Age, Gender, Study Description, Upload Date, Accession Number,
        Abnormal, Atelectasis, Calcification, Cardiomegaly, Consolidation, Fibrosis, Mediastinal Widening, Nodule, Pleural Effusion, Pneumoperitoneum, Pneumothorax, Tuberculosis, 
        Feedback, Comments, Status, Inference Date, AI Report, WORKPLACE, PROCEDURE_START_DATE, PROCEDURE_END_DATE, AI_PRIORITY, AI_FLAG_RECEIVED_DATE, TAT, Time_to_clinical_decision(mins),
        TEXT_REPORT, ground truth, llm_grade_binary, Overall_binary, tb]
        
        We need to rename some of the columns:
        - 'ground truth' to 'gt_manual'
        - 'llm_grade_binary' to 'gt_llm'
        - 'Overall_binary' to 'lunit_binarised'
        '''
        
        # Rename columns
        df = df.rename(columns={
            'ground truth': 'gt_manual',
            'llm_grade_binary': 'gt_llm',
            'Overall_binary': 'lunit_binarised',
            'Time_to_Clinical_Decision': 'Time to clinical_decision(mins)'
        })
        
        # Drop unnecessary columns
        columns_to_drop_final = [
            'PROCEDURE_CODE', 'MEDICAL_LOCATION_NAME', 'Age', 'Gender', 'Accession Number'
        ]
        df = df.drop(columns=columns_to_drop_final, errors='ignore')

        # Define desired column order
        desired_columns = [
            'StudyIDAnonymized', 'StudyID', 'Patient Name', 'Patient ID', 'Instances', 
            'PATIENT_AGE', 'PATIENT_GENDER', 'Study Description', 'Upload Date', 'ACCESSION_NO',
            'Abnormal', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 
            'Fibrosis', 'Mediastinal Widening', 'Nodule', 'Pleural Effusion', 
            'Pneumoperitoneum', 'Pneumothorax', 'Tuberculosis',
            'Feedback', 'Comments', 'Status', 'Inference Date', 'AI Report', 
            'WORKPLACE', 'PROCEDURE_START_DATE', 'PROCEDURE_END_DATE', 'AI_PRIORITY', 
            'AI_FLAG_RECEIVED_DATE', 'Time_End_to_End', 'Time to clinical_decision(mins)',
            'TEXT_REPORT', 'gt_manual', 'gt_llm', 'lunit_binarised'
        ]

        # Filter to only include columns that exist in the dataframe
        available_columns = [col for col in desired_columns if col in df.columns]

        # Reorder columns and add any remaining columns at the end
        remaining_columns = [col for col in df.columns if col not in available_columns]
        df = df[available_columns + remaining_columns]

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