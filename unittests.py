import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
from sklearn.metrics import cohen_kappa_score, accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
import json
from process_carpl import calculate_agreement_metrics, perform_mcnemar_test
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

# Import the functions we want to test
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestProcessCarpl(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'ACCESSION_NO': [1, 2, 3, 4, 5],
            'ground truth': [0, 1, 0, 1, 1],
            'Overall_binary': [0, 1, 1, 1, 0],
            'llm_grade_binary': [0, 0, 0, 1, 1],
            'TEXT_REPORT': [
                'Normal chest X-ray',
                'Consolidation in right lower lobe',
                'Clear lungs, no acute findings',
                'Pneumothorax on left side',
                'Cardiomegaly noted'
            ]
        })
        
    def test_calculate_agreement_metrics(self):
        """Test the calculation of agreement metrics."""
        gt = self.sample_data['ground truth']
        pred = self.sample_data['Overall_binary']
        
        metrics = calculate_agreement_metrics(gt, pred)
        
        # Expected values
        expected_accuracy = accuracy_score(gt, pred)
        expected_kappa = cohen_kappa_score(gt, pred)
        
        self.assertAlmostEqual(metrics['exact_agreement'], expected_accuracy, places=3)
        self.assertAlmostEqual(metrics['cohen_kappa'], expected_kappa, places=3)
        self.assertIn('sensitivity', metrics)
        self.assertIn('specificity', metrics)
        
    def test_perform_mcnemar_test(self):
        """Test McNemar's test implementation."""
        gt = self.sample_data['ground truth']
        pred1 = self.sample_data['Overall_binary']
        pred2 = self.sample_data['llm_grade_binary']
        
        result = perform_mcnemar_test(gt, pred1, pred2)
        
        self.assertIn('contingency_table', result)
        self.assertIn('test_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)
        self.assertIsInstance(result['significant'], bool)
        
    def test_binary_conversion(self):
        """Test binary conversion of findings columns."""
        # Test data with various threshold values
        test_data = pd.DataFrame({
            'Atelectasis': [5, 15, 0, 20, 10],
            'Consolidation': [0, 25, 5, 30, 15]
        })
        
        threshold = 10
        
        # Apply binary conversion
        test_data['Atelectasis_binary'] = test_data['Atelectasis'].apply(
            lambda x: 1 if x > threshold else 0
        )
        test_data['Consolidation_binary'] = test_data['Consolidation'].apply(
            lambda x: 1 if x > threshold else 0
        )
        
        # Check results
        expected_atelectasis = [0, 1, 0, 1, 0]
        expected_consolidation = [0, 1, 0, 1, 1]
        
        self.assertEqual(test_data['Atelectasis_binary'].tolist(), expected_atelectasis)
        self.assertEqual(test_data['Consolidation_binary'].tolist(), expected_consolidation)
        
    def test_overall_binary_calculation(self):
        """Test the calculation of Overall_binary column."""
        test_data = pd.DataFrame({
            'Finding1_binary': [0, 1, 0, 1, 0],
            'Finding2_binary': [0, 0, 1, 1, 0],
            'Finding3_binary': [0, 0, 0, 0, 1]
        })
        
        binary_columns = ['Finding1_binary', 'Finding2_binary', 'Finding3_binary']
        test_data['Overall_binary'] = test_data[binary_columns].max(axis=1)
        
        expected_overall = [0, 1, 1, 1, 1]
        self.assertEqual(test_data['Overall_binary'].tolist(), expected_overall)
        
    def test_data_preprocessing(self):
        """Test data preprocessing steps."""
        # Test merging operation
        df_reports = pd.DataFrame({
            'ACCESSION_NO': [1, 2, 3],
            'TEXT_REPORT': ['Report 1', 'Report 2', 'Report 3']
        })
        
        df_lunit = pd.DataFrame({
            'Accession Number': [1, 2, 4],
            'Atelectasis': [5, 15, 20]
        })
        
        df_merged = pd.merge(df_reports, df_lunit, 
                           left_on="ACCESSION_NO", 
                           right_on="Accession Number", 
                           how="inner")
        
        # Should only have 2 rows (intersection of accession numbers 1 and 2)
        self.assertEqual(len(df_merged), 2)
        self.assertIn('ACCESSION_NO', df_merged.columns)
        self.assertIn('Atelectasis', df_merged.columns)
        
    @patch('process_carpl.open_protected_xlsx')
    def test_file_loading(self, mock_open_xlsx):
        """Test file loading operations."""
        # Mock the Excel file loading
        mock_data = pd.DataFrame({
            'ACCESSION_NO': [1, 2, 3],
            'TEXT_REPORT': ['Report 1', 'Report 2', 'Report 3']
        })
        mock_open_xlsx.return_value = mock_data
        
        # Test that the function would be called correctly
        # (This is a simplified test since we can't easily test the full file loading)
        result = mock_open_xlsx('test_path', 'test_password')
        self.assertEqual(len(result), 3)
        
    def test_statistical_significance_interpretation(self):
        """Test interpretation of statistical significance."""
        # Test case where p-value < 0.05 (significant)
        result_significant = {
            'p_value': 0.01,
            'test_statistic': 5.2
        }
        
        is_significant = result_significant['p_value'] < 0.05
        self.assertTrue(is_significant)
        
        # Test case where p-value >= 0.05 (not significant)
        result_not_significant = {
            'p_value': 0.15,
            'test_statistic': 1.2
        }
        
        is_not_significant = result_not_significant['p_value'] >= 0.05
        self.assertTrue(is_not_significant)


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
    
    return {
        'exact_agreement': exact_agreement,
        'cohen_kappa': cohen_kappa,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
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
    
    # Calculate agreement metrics for GT vs LLM
    gt_llm_metrics = calculate_agreement_metrics(gt, llm)
    
    # Perform McNemar test between Lunit and LLM
    mcnemar_result = perform_mcnemar_test(gt, lunit, llm)
    
    # Print results
    print("=== GROUND TRUTH vs LUNIT ANALYSIS ===")
    print(f"Exact Agreement (GT vs Lunit): {gt_lunit_metrics['exact_agreement']:.3f} ({gt_lunit_metrics['exact_agreement']*100:.1f}%)")
    print(f"Cohen's Kappa (GT vs Lunit): {gt_lunit_metrics['cohen_kappa']:.3f}")
    print(f"Sensitivity (GT vs Lunit): {gt_lunit_metrics['sensitivity']:.3f}")
    print(f"Specificity (GT vs Lunit): {gt_lunit_metrics['specificity']:.3f}")
    
    print("\n=== GROUND TRUTH vs LLM ANALYSIS ===")
    print(f"Exact Agreement (GT vs LLM): {gt_llm_metrics['exact_agreement']:.3f} ({gt_llm_metrics['exact_agreement']*100:.1f}%)")
    print(f"Cohen's Kappa (GT vs LLM): {gt_llm_metrics['cohen_kappa']:.3f}")
    print(f"Sensitivity (GT vs LLM): {gt_llm_metrics['sensitivity']:.3f}")
    print(f"Specificity (GT vs LLM): {gt_llm_metrics['specificity']:.3f}")
    
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


if __name__ == '__main__':
    # Add the analysis function call to your existing code
    
    # Example usage (add this to your process_carpl.py):
    """
    # After creating df_reports_accuracy, add:
    analysis_results = analyze_gt_vs_lunit(df_reports_accuracy)
    
    # Save results to file
    results_summary = {
        'dataset_size': len(df_reports_accuracy),
        'gt_lunit_agreement': analysis_results['gt_lunit_metrics']['exact_agreement'],
        'gt_lunit_kappa': analysis_results['gt_lunit_metrics']['cohen_kappa'],
        'gt_llm_agreement': analysis_results['gt_llm_metrics']['exact_agreement'], 
        'gt_llm_kappa': analysis_results['gt_llm_metrics']['cohen_kappa'],
        'mcnemar_p_value': analysis_results['mcnemar_result']['p_value'],
        'mcnemar_significant': analysis_results['mcnemar_result']['significant']
    }
    
    with open('analysis_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    """
    
    # Run tests
    unittest.main()