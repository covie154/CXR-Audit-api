#%% 
import pandas as pd
import numpy as np
import os
import re
import json
from telegram_notifier import Notifier
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

# Change to the scripts_lunit_review directory
#os.chdir('scripts_lunit_review')
#print(f"New working directory: {os.getcwd()}")

from open_protected_xlsx import open_protected_xlsx
from cxr_audit.lib_audit_cxr_v2 import CXRClassifier
from cxr_audit.grade_batch_async import BatchCXRProcessor

# Telegram notifier
bot_token = '1039031996:AAGu-1B3kUi-pJcDD-1SeARze-oBxMmdp-k'
chat_id = '-410738508'

# Temporarily set the TELEGRAM_CHAT_ID environment variable
os.environ["TELEGRAM_CHAT_ID"] = chat_id
os.environ["TELEGRAM_TOKEN"] = bot_token
telegram_notifier = Notifier()

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

# Get the current working directory and navigate to data_audit folder
current_dir = os.getcwd()
base_path = os.path.abspath(os.path.join(current_dir, '..', 'data_lunit_review'))
data_file = 'deployment_stats14.csv'
df_lunit = pd.read_csv(os.path.join(base_path, data_file))

findings_columns = ["Atelectasis", "Calcification", "Cardiomegaly", \
                    "Consolidation", "Fibrosis", "Mediastinal Widening", \
                    "Nodule", "Pleural Effusion", "Pneumoperitoneum", \
                    "Pneumothorax", "Tuberculosis"]

path_reports = "../data_lunit_review/07-Aug-2025/07-Aug-2025"
reports_file = "RIS_WeeklyReport_07Aug2025.xls"
reports_path = os.path.join(path_reports, reports_file)
df_reports = open_protected_xlsx(reports_path, "GE_2024_P@55")
print("")

df_merged = pd.merge(df_reports, df_lunit, left_on="ACCESSION_NO", right_on="Accession Number", how="inner")
# Drop specified columns from df_merged
# There are basically two parts to this exercise:
# 1. Accuracy audit
# 2. Time audit
columns_to_drop = ['MEDICAL_LOCATION_CODE', 'PROCEDURE_NAME']
df_merged = df_merged.drop(columns=columns_to_drop, errors='ignore')

# Determine the first and last date
first_date = df_merged['PROCEDURE_START_DATE'].min()
first_date_formatted = first_date.strftime("%d-%b-%Y")
last_date = df_merged['PROCEDURE_START_DATE'].max()
last_date_formatted = last_date.strftime("%d-%b-%Y")
# Calculate the number of Non-CHESTs inferenced
non_chest_count = len(df_merged[df_merged['PROCEDURE_CODE'] != 556])
# Determine the total number of sites inferenced from
sites_count = df_merged.groupby('MEDICAL_LOCATION_NAME').size()

print(f"# REVIEW FOR PERIOD: {first_date_formatted} to {last_date_formatted}")
print(f"Number of studies inferenced: {len(df_merged)}")
print("")
print("# BASIC DESCRIPTORS")
print(f"Number of inappropriate studies inferenced: {non_chest_count}")
print("  (Inappropriate studies include: Chest (Screening), AP/Oblique, AP/Lateral)")
print("")
print("Studies were inferenced from the following sites:")
for site, count in sites_count.items():
    print(f" - {site}: {count}")


# %%
# Now we can start processing proper

priority_threshold = 10

for column in findings_columns:
    new_column_name = column + "_binary"
    df_merged[new_column_name] = df_merged[column].apply(lambda x: 1 if x > priority_threshold else 0)

# Create Overall_binary column
binary_columns = [col + "_binary" for col in findings_columns]
df_merged['Overall_binary'] = df_merged[binary_columns].max(axis=1)

# Compute LLM score and binarize it
processed_reports = processor.process_full_pipeline(df_merged, report_column='TEXT_REPORT', steps=['llm'])
processed_reports['llm_grade_binary'] = processed_reports['llm_grade'].apply(lambda x: 1 if x > 2 else 0)

# %%
# Now, let's see some basic statistics

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


def analyze_gt_vs_lunit(df_reports_accuracy):
    """
    Analyze agreement between ground truth and Lunit predictions.
    
    Args:
        df_reports_accuracy (pd.DataFrame): DataFrame with columns 'gt' and 'lunit'
        
    Returns:
        dict: Comprehensive analysis results
    """
    gt = df_reports_accuracy['gt']
    lunit = df_reports_accuracy['lunit']
    llm = df_reports_accuracy['llm']
    
    # Calculate agreement metrics for GT vs Lunit
    gt_lunit_metrics = calculate_agreement_metrics(gt, lunit)
    
    # Calculate agreement metrics for LLM vs Lunit
    llm_lunit_metrics = calculate_agreement_metrics(llm, lunit)

    # Calculate agreement metrics for GT vs LLM
    gt_llm_metrics = calculate_agreement_metrics(gt, llm)

    # Perform McNemar test between gt and LLM
    mcnemar_result = perform_mcnemar_test(lunit, gt, llm)
    
    # Print results
    print("=== GROUND TRUTH vs LUNIT ANALYSIS (ORIGINAL METHOD) ===")
    print(f"Exact Agreement (GT vs Lunit): {gt_lunit_metrics['exact_agreement']:.3f} ({gt_lunit_metrics['exact_agreement']*100:.1f}%)")
    print(f"Cohen's Kappa (GT vs Lunit): {gt_lunit_metrics['cohen_kappa']:.3f}")
    print(f"Sensitivity (GT vs Lunit): {gt_lunit_metrics['sensitivity']:.3f}")
    print(f"Specificity (GT vs Lunit): {gt_lunit_metrics['specificity']:.3f}")
    print(f"PPV (GT vs Lunit): {gt_lunit_metrics['positive_predictive_value']:.3f}")
    print(f"NPV (GT vs Lunit): {gt_lunit_metrics['negative_predictive_value']:.3f}")

    print("\n=== LLM vs LUNIT ANALYSIS (NEW METHOD) ===")
    print(f"Exact Agreement (LLM vs Lunit): {llm_lunit_metrics['exact_agreement']:.3f} ({llm_lunit_metrics['exact_agreement']*100:.1f}%)")
    print(f"Cohen's Kappa (LLM vs Lunit): {llm_lunit_metrics['cohen_kappa']:.3f}")
    print(f"Sensitivity (LLM vs Lunit): {llm_lunit_metrics['sensitivity']:.3f}")
    print(f"Specificity (LLM vs Lunit): {llm_lunit_metrics['specificity']:.3f}")
    print(f"PPV (LLM vs Lunit): {llm_lunit_metrics['positive_predictive_value']:.3f}")
    print(f"NPV (LLM vs Lunit): {llm_lunit_metrics['negative_predictive_value']:.3f}")

    print("\n=== GROUND TRUTH vs LLM ANALYSIS (VALIDATION)===")
    print(f"Exact Agreement (GT vs LLM): {gt_llm_metrics['exact_agreement']:.3f} ({gt_llm_metrics['exact_agreement']*100:.1f}%)")
    print(f"Cohen's Kappa (GT vs LLM): {gt_llm_metrics['cohen_kappa']:.3f}")
    
    print("\n=== McNEMAR'S TEST (Lunit vs LLM) ===")
    print("Contingency Table (Lunit rows, LLM columns):")
    print("0 = Incorrect, 1 = Correct")
    print(mcnemar_result['contingency_table'])
    print(f"\nTest Statistic: {mcnemar_result['test_statistic']:.3f}")
    print(f"P-value: {mcnemar_result['p_value']:.6f}")
    print(f"Statistically Significant: {'Yes' if mcnemar_result['significant'] else 'No'}")
    print(f"Accuracy Difference (LLM - Lunit): {mcnemar_result['accuracy_difference']:.3f}")
    
    return {
        'gt_lunit_metrics': gt_lunit_metrics,
        'gt_llm_metrics': gt_llm_metrics,
        'mcnemar_result': mcnemar_result
    }
    
df_reports_accuracy = processed_reports.copy()[['ACCESSION_NO', 'ground truth', 'Overall_binary', 'llm_grade_binary']]
df_reports_accuracy.columns = ['accession', 'gt', 'lunit', 'llm']
stats_results = analyze_gt_vs_lunit(df_reports_accuracy)

# %%
# Time analysis

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
    
def convert_to_minutes(td):
    if pd.isnull(td):
        return np.nan
    time_mins = td.total_seconds() / 60
    time_secs = td.total_seconds() % 60
    return f"{int(time_mins)}m {int(time_secs)}s"
    
df_merged['Time_to_Clinical_Decision'] = df_merged.apply(time_clinical_decision, axis=1)
df_merged['Time_End_to_End'] = df_merged.apply(time_end_to_end, axis=1)
# Convert float minutes to pandas.Timedelta
df_merged['Time_to_Clinical_Decision'] = pd.to_timedelta(df_merged['Time_to_Clinical_Decision'], unit='seconds')
df_merged['Time_End_to_End'] = pd.to_timedelta(df_merged['Time_End_to_End'], unit='seconds')

# Create box plots for time analysis
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

# Print summary statistics
print("\n=== TIME ANALYSIS SUMMARY ===")
print("Time to Clinical Decision:")
print("(time from Exam End to AI Flag Received (i.e. case processed), otherwise Report TAT (i.e. case not processed))")
print(f"  Mean: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].mean())}")
print(f"  Median: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].median())}")
print(f"  Std: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].std())}")
print(f"  Min: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].min())}")
print(f"  Max: {convert_to_minutes(df_merged['Time_to_Clinical_Decision'].max())}")

print("\nEnd to End Server Time:")
print("(time from Exam End to AI Flag Received, excluding cases where the flag never made it)")
print(f"  Mean: {convert_to_minutes(time_end_to_end_nonzero.mean())}")
print(f"  Median: {convert_to_minutes(time_end_to_end_nonzero.median())}")
print(f"  Std: {convert_to_minutes(time_end_to_end_nonzero.std())}")
print(f"  Min: {convert_to_minutes(time_end_to_end_nonzero.min())}")
print(f"  Max: {convert_to_minutes(time_end_to_end_nonzero.max())}")

cases_late = len(time_end_to_end_nonzero[time_end_to_end_nonzero > pd.Timedelta(minutes=5)])
total_len = len(df_merged)
print(f"  % of cases with t > 5 mins: {cases_late / total_len * 100:.2f}% ({cases_late} / {total_len} case(s))")
#%%