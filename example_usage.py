#%%
"""
Example usage of the password-protected Excel file reader
"""

from open_protected_xlsx import open_protected_xlsx
import os

def example_usage():
    """
    Example of how to use the open_protected_xlsx function
    """
    
    # Example 1: Using the function directly
    print("Example 1: Direct function usage")
    print("=" * 40)
    
    # Replace these with your actual file path and password
    file_path = "your_protected_file.xlsx"
    password = "your_password"
    
    # Uncomment the lines below when you have an actual protected file to test
    # df = open_protected_xlsx(file_path, password)
    # if df is not None:
    #     print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    #     print(df.head())
    
    print("\nTo use this script:")
    print("1. Replace 'your_protected_file.xlsx' with the actual file path")
    print("2. Replace 'your_password' with the actual password")
    print("3. Run the script")
    
    print("\nAlternatively, you can run the main script with command line arguments:")
    print("python open_protected_xlsx.py <file_path> <password> [sheet_name] [output_csv_path]")
    
    print("\nOr run it interactively:")
    print("python open_protected_xlsx.py")

if __name__ == "__main__":
    example_usage()

# %%
file_path = r"C:\Users\Covie\OneDrive\Documents\Work\Research\PRIME\data_lunit_review\07-Aug-2025\07-Aug-2025\RIS_WeeklyReport_07Aug2025.xls"
open_protected_xlsx(file_path, "GE_2024_P@55")
# %%
