#!/usr/bin/env python3
"""
Example script showing how to use ProcessCarpl with multiple input files.

This demonstrates the updated functionality that supports combining multiple 
CARPL (deployment stats) files and multiple GE (RIS report) files.
"""

from class_process_carpl import ProcessCarpl
import os

def example_single_files():
    """Example using single files (backward compatible)"""
    print("=== Example 1: Single Files (Backward Compatible) ===")
    
    # Single file paths
    carpl_file = "deployment_stats14.csv"
    ge_file = "RIS_WeeklyReport_example.xlsx"
    
    # This still works as before
    processor = ProcessCarpl(carpl_file, ge_file)
    df_merged = processor.load_reports()
    
    print(f"Processed {len(df_merged)} merged records from single files")
    return df_merged

def example_multiple_files():
    """Example using multiple files"""
    print("\n=== Example 2: Multiple Files ===")
    
    # Multiple file paths for CARPL reports
    carpl_files = [
        "deployment_stats14.csv",
        "deployment_stats15.csv",
        "deployment_stats16.csv"
    ]
    
    # Multiple file paths for GE reports
    ge_files = [
        "RIS_WeeklyReport_week1.xlsx",
        "RIS_WeeklyReport_week2.xlsx",
        "RIS_WeeklyReport_week3.xlsx"
    ]
    
    # Create processor with multiple files
    processor = ProcessCarpl(carpl_files, ge_files)
    df_merged = processor.load_reports()
    
    print(f"Processed {len(df_merged)} merged records from multiple files")
    return df_merged

def example_mixed_file_types():
    """Example using mixed file types"""
    print("\n=== Example 3: Mixed File Types ===")
    
    # Mix of CSV and Excel files for CARPL
    carpl_files = [
        "deployment_stats.csv",
        "deployment_stats.xlsx"
    ]
    
    # Mix of CSV and Excel files for GE
    ge_files = [
        "ris_report.csv",
        "ris_report.xlsx"
    ]
    
    processor = ProcessCarpl(carpl_files, ge_files)
    df_merged = processor.load_reports()
    
    print(f"Processed {len(df_merged)} merged records from mixed file types")
    return df_merged

def example_with_custom_settings():
    """Example with custom processor settings"""
    print("\n=== Example 4: Custom Settings ===")
    
    carpl_files = ["deployment_stats14.csv"]
    ge_files = ["RIS_WeeklyReport.xlsx"]
    
    # Custom settings
    processor = ProcessCarpl(
        path_carpl_reports=carpl_files,
        path_ge_reports=ge_files,
        priority_threshold=15,  # Custom threshold
        passwd="custom_password_2024"  # Custom password
    )
    
    df_merged = processor.load_reports()
    
    print(f"Processed {len(df_merged)} merged records with custom settings")
    return df_merged

def example_full_pipeline():
    """Example running the full analysis pipeline"""
    print("\n=== Example 5: Full Pipeline ===")
    
    carpl_files = [
        "deployment_stats14.csv",
        "deployment_stats15.csv"
    ]
    
    ge_files = [
        "RIS_WeeklyReport_week1.xlsx",
        "RIS_WeeklyReport_week2.xlsx"
    ]
    
    processor = ProcessCarpl(carpl_files, ge_files)
    
    # Run the full pipeline
    df_merged = processor.load_reports()
    
    # Generate text reports
    initial_metrics = processor.txt_initial_metrics(df_merged)
    print("Initial Metrics:")
    print(initial_metrics)
    
    # Process accuracy statistics
    df_merged = processor.process_stats_accuracy(df_merged)
    stats_accuracy = processor.txt_stats_accuracy(df_merged)
    print("\nAccuracy Statistics:")
    print(stats_accuracy)
    
    # Process time statistics
    df_merged = processor.process_stats_time(df_merged)
    stats_time = processor.txt_stats_time(df_merged)
    print("\nTime Statistics:")
    print(stats_time)
    
    # Rearrange columns
    df_merged = processor.rearrange_columns(df_merged)
    
    # Save results
    output_filename = f"multiple_files_analysis_results.csv"
    df_merged.to_csv(output_filename, index=False)
    print(f"\nResults saved to: {output_filename}")
    
    return df_merged

if __name__ == "__main__":
    print("ProcessCarpl Multiple Files Examples")
    print("=" * 50)
    
    # Note: Uncomment the examples you want to run
    # Make sure the file paths exist in your environment
    
    # Example 1: Single files (backward compatible)
    # example_single_files()
    
    # Example 2: Multiple files
    # example_multiple_files()
    
    # Example 3: Mixed file types
    # example_mixed_file_types()
    
    # Example 4: Custom settings
    # example_with_custom_settings()
    
    # Example 5: Full pipeline
    # example_full_pipeline()
    
    print("\nTo run these examples:")
    print("1. Update file paths to match your actual files")
    print("2. Uncomment the example functions you want to test")
    print("3. Run this script")