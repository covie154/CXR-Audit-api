# %%
# Finding the optimal threshold for Lunit scores
## EXPERIMENTAL
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def find_optimal_threshold(df_merged, findings_columns, ground_truth_col='ground truth', 
                          sensitivity_weight=0.7, specificity_weight=0.3, plot=True):
    """
    Find optimal threshold for binarizing Lunit scores to maximize weighted sensitivity and specificity.
    
    Args:
        df_merged (pd.DataFrame): DataFrame containing Lunit scores and ground truth
        findings_columns (list): List of finding column names
        ground_truth_col (str): Name of ground truth column
        sensitivity_weight (float): Weight for sensitivity (0-1, higher favors sensitivity)
        specificity_weight (float): Weight for specificity (0-1, should sum to 1 with sensitivity_weight)
        plot (bool): Whether to create plots
        
    Returns:
        dict: Dictionary containing optimal threshold and metrics
    """
    
    # Get ground truth
    ground_truth = df_merged[ground_truth_col]
    
    # Test range of thresholds
    thresholds = np.arange(0, 100, 1)
    results = []
    
    for threshold in thresholds:
        # Create binary predictions for each finding
        binary_predictions = {}
        for column in findings_columns:
            binary_predictions[column] = df_merged[column].apply(lambda x: 1 if x > threshold else 0)
        
        # Create overall binary prediction (max across all findings)
        overall_binary = pd.DataFrame(binary_predictions).max(axis=1)
        
        # Calculate metrics
        metrics = calculate_agreement_metrics(ground_truth, overall_binary)
        
        # Calculate weighted score (favoring sensitivity)
        weighted_score = (sensitivity_weight * metrics['sensitivity'] + 
                         specificity_weight * metrics['specificity'])
        
        results.append({
            'threshold': threshold,
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'weighted_score': weighted_score,
            'exact_agreement': metrics['exact_agreement'],
            'cohen_kappa': metrics['cohen_kappa'],
            'true_positives': metrics['true_positives'],
            'true_negatives': metrics['true_negatives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives']
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold
    optimal_idx = results_df['weighted_score'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    optimal_metrics = results_df.loc[optimal_idx]
    
    if plot:
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Sensitivity and Specificity vs Threshold
        axes[0, 0].plot(results_df['threshold'], results_df['sensitivity'], 'b-', label='Sensitivity', linewidth=2)
        axes[0, 0].plot(results_df['threshold'], results_df['specificity'], 'r-', label='Specificity', linewidth=2)
        axes[0, 0].plot(results_df['threshold'], results_df['weighted_score'], 'g--', 
                       label=f'Weighted Score (Sen={sensitivity_weight}, Spec={specificity_weight})', linewidth=2)
        axes[0, 0].axvline(x=optimal_threshold, color='black', linestyle=':', 
                          label=f'Optimal Threshold = {optimal_threshold}')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Sensitivity, Specificity, and Weighted Score vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Agreement metrics vs Threshold
        axes[0, 1].plot(results_df['threshold'], results_df['exact_agreement'], 'purple', 
                       label='Exact Agreement', linewidth=2)
        axes[0, 1].plot(results_df['threshold'], results_df['cohen_kappa'], 'orange', 
                       label="Cohen's Kappa", linewidth=2)
        axes[0, 1].axvline(x=optimal_threshold, color='black', linestyle=':', 
                          label=f'Optimal Threshold = {optimal_threshold}')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Agreement Metrics vs Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: ROC-like curve (Sensitivity vs 1-Specificity)
        axes[1, 0].plot(1 - results_df['specificity'], results_df['sensitivity'], 'b-', linewidth=2)
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        # Mark optimal point
        opt_sens = optimal_metrics['sensitivity']
        opt_spec = optimal_metrics['specificity']
        axes[1, 0].plot(1 - opt_spec, opt_sens, 'ro', markersize=10, 
                       label=f'Optimal (Threshold={optimal_threshold})')
        
        axes[1, 0].set_xlabel('1 - Specificity (False Positive Rate)')
        axes[1, 0].set_ylabel('Sensitivity (True Positive Rate)')
        axes[1, 0].set_title('ROC-like Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Confusion matrix components
        axes[1, 1].plot(results_df['threshold'], results_df['true_positives'], 'g-', 
                       label='True Positives', linewidth=2)
        axes[1, 1].plot(results_df['threshold'], results_df['true_negatives'], 'b-', 
                       label='True Negatives', linewidth=2)
        axes[1, 1].plot(results_df['threshold'], results_df['false_positives'], 'r-', 
                       label='False Positives', linewidth=2)
        axes[1, 1].plot(results_df['threshold'], results_df['false_negatives'], 'orange', 
                       label='False Negatives', linewidth=2)
        axes[1, 1].axvline(x=optimal_threshold, color='black', linestyle=':', 
                          label=f'Optimal Threshold = {optimal_threshold}')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Confusion Matrix Components vs Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Detailed view around optimal threshold
        fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Focus on range around optimal threshold
        focus_range = 20
        min_thresh = max(0, optimal_threshold - focus_range)
        max_thresh = min(99, optimal_threshold + focus_range)
        
        mask = (results_df['threshold'] >= min_thresh) & (results_df['threshold'] <= max_thresh)
        focused_results = results_df[mask]
        
        ax.plot(focused_results['threshold'], focused_results['sensitivity'], 'b-', 
               label='Sensitivity', linewidth=3, marker='o', markersize=4)
        ax.plot(focused_results['threshold'], focused_results['specificity'], 'r-', 
               label='Specificity', linewidth=3, marker='s', markersize=4)
        ax.plot(focused_results['threshold'], focused_results['weighted_score'], 'g--', 
               label=f'Weighted Score', linewidth=3, marker='^', markersize=4)
        
        ax.axvline(x=optimal_threshold, color='black', linestyle=':', linewidth=2,
                  label=f'Optimal Threshold = {optimal_threshold}')
        
        # Add text annotation for optimal point
        ax.annotate(f'Optimal: {optimal_threshold}\nSen: {opt_sens:.3f}\nSpec: {opt_spec:.3f}',
                   xy=(optimal_threshold, optimal_metrics['weighted_score']),
                   xytext=(optimal_threshold + 5, optimal_metrics['weighted_score'] + 0.05),
                   arrowprops=dict(arrowstyle='->', color='black'),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Detailed View: Optimal Threshold Analysis\n(Sensitivity Weight: {sensitivity_weight}, Specificity Weight: {specificity_weight})', 
                    fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Print summary
    print(f"\n=== OPTIMAL THRESHOLD ANALYSIS ===")
    print(f"Optimal Threshold: {optimal_threshold}")
    print(f"Sensitivity Weight: {sensitivity_weight}")
    print(f"Specificity Weight: {specificity_weight}")
    print(f"\nOptimal Metrics:")
    print(f"  Sensitivity: {optimal_metrics['sensitivity']:.3f}")
    print(f"  Specificity: {optimal_metrics['specificity']:.3f}")
    print(f"  Weighted Score: {optimal_metrics['weighted_score']:.3f}")
    print(f"  Exact Agreement: {optimal_metrics['exact_agreement']:.3f}")
    print(f"  Cohen's Kappa: {optimal_metrics['cohen_kappa']:.3f}")
    print(f"  True Positives: {optimal_metrics['true_positives']}")
    print(f"  True Negatives: {optimal_metrics['true_negatives']}")
    print(f"  False Positives: {optimal_metrics['false_positives']}")
    print(f"  False Negatives: {optimal_metrics['false_negatives']}")
    
    return {
        'optimal_threshold': optimal_threshold,
        'optimal_metrics': optimal_metrics,
        'all_results': results_df,
        'sensitivity_weight': sensitivity_weight,
        'specificity_weight': specificity_weight
    }

"""
# Test different weighting schemes
print("=== TESTING DIFFERENT SENSITIVITY/SPECIFICITY WEIGHTS ===")

# Heavily favor sensitivity (70/30)
optimal_results_70_30 = find_optimal_threshold(df_merged, findings_columns, 
                                               sensitivity_weight=0.7, specificity_weight=0.3)

# Moderately favor sensitivity (60/40)
optimal_results_60_40 = find_optimal_threshold(df_merged, findings_columns, 
                                               sensitivity_weight=0.6, specificity_weight=0.4, plot=False)

# Equal weighting (50/50)
optimal_results_50_50 = find_optimal_threshold(df_merged, findings_columns, 
                                               sensitivity_weight=0.5, specificity_weight=0.5, plot=False)

# Compare results
comparison_results = pd.DataFrame({
    'Weight_Scheme': ['70/30 (Favor Sen)', '60/40 (Favor Sen)', '50/50 (Equal)'],
    'Optimal_Threshold': [optimal_results_70_30['optimal_threshold'], 
                         optimal_results_60_40['optimal_threshold'],
                         optimal_results_50_50['optimal_threshold']],
    'Sensitivity': [optimal_results_70_30['optimal_metrics']['sensitivity'],
                   optimal_results_60_40['optimal_metrics']['sensitivity'],
                   optimal_results_50_50['optimal_metrics']['sensitivity']],
    'Specificity': [optimal_results_70_30['optimal_metrics']['specificity'],
                   optimal_results_60_40['optimal_metrics']['specificity'],
                   optimal_results_50_50['optimal_metrics']['specificity']],
    'Weighted_Score': [optimal_results_70_30['optimal_metrics']['weighted_score'],
                      optimal_results_60_40['optimal_metrics']['weighted_score'],
                      optimal_results_50_50['optimal_metrics']['weighted_score']]
})

print("\n=== COMPARISON OF DIFFERENT WEIGHTING SCHEMES ===")
print(comparison_results)

# Use the 70/30 result as the recommended threshold
recommended_threshold = optimal_results_70_30['optimal_threshold']
print(f"\n=== RECOMMENDATION ===")
print(f"Recommended threshold (favoring sensitivity): {recommended_threshold}")

# Update the priority_threshold variable
#priority_threshold = recommended_threshold

# Recalculate binary columns with optimal threshold
#for column in findings_columns:
#    new_column_name = column + "_binary"
#    df_merged[new_column_name] = df_merged[column].apply(lambda x: 1 if x > priority_threshold else 0)

# Update Overall_binary column
#binary_columns = [col + "_binary" for col in findings_columns]
#df_merged['Overall_binary'] = df_merged[binary_columns].max(axis=1)

print(f"\nUpdated binary predictions using optimal threshold: {priority_threshold}")
"""