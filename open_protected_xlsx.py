"""
Script to open a password-protected Excel file, convert to DataFrame, and save as CSV
"""

import pandas as pd
import openpyxl
from msoffcrypto import OfficeFile
import io
import os
import sys

def open_protected_xlsx(file_path, password, sheet_name=None, output_csv=False, output_csv_path=None):
    """
    Open a password-protected Excel file and convert to DataFrame
    
    Parameters:
    file_path (str): Path to the password-protected Excel file
    password (str): Password for the Excel file
    sheet_name (str or int, optional): Name or index of the sheet to read. If None, reads the first sheet
    output_csv (bool, optional): Save CSV file?
    output_csv_path (str, optional): Path to save the CSV file. If None, uses same name as Excel file
    
    Returns:
    pandas.DataFrame: DataFrame containing the Excel data
    """
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Opening password-protected file: {file_path}")
        
        decrypted_content = io.BytesIO()
        
        # Open the encrypted file
        with open(file_path, 'rb') as file:
            office_file = OfficeFile(file)
            office_file.load_key(password=password)
            office_file.decrypt(decrypted_content)
        
        # Read the decrypted Excel file into a DataFrame
        print("Converting to DataFrame...")
        if sheet_name is not None:
            df = pd.read_excel(decrypted_content, sheet_name=sheet_name)
        else:
            df = pd.read_excel(decrypted_content)
        
        print(f"Successfully loaded DataFrame with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Save the CSV if we want to
        if output_csv:
            # Generate output CSV path if not provided
            if output_csv_path is None:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_dir = os.path.dirname(file_path)
                output_csv_path = os.path.join(output_dir, f"{base_name}.csv")
            
            # Save to CSV
            print(f"Saving to CSV: {output_csv_path}")
            df.to_csv(output_csv_path, index=False)
            print("CSV file saved successfully!")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")

def main():
    """
    Main function to run the script with command line arguments or interactive input
    """
    
    # Check if arguments provided via command line
    if len(sys.argv) >= 3:
        file_path = sys.argv[1]
        password = sys.argv[2]
        sheet_name = sys.argv[3] if len(sys.argv) > 3 else None
        output_csv_path = sys.argv[4] if len(sys.argv) > 4 else None
    else:
        # Interactive input
        print("Password-Protected Excel to CSV Converter")
        print("=" * 40)
        
        file_path = input("Enter the path to the Excel file: ").strip()
        password = input("Enter the password: ").strip()
        
        sheet_input = input("Enter sheet name/index (press Enter for first sheet): ").strip()
        sheet_name = sheet_input if sheet_input else None
        
        output_input = input("Enter output CSV path (press Enter for auto-generate): ").strip()
        output_csv_path = output_input if output_input else None
    
    # Process the file
    df = open_protected_xlsx(file_path, password, sheet_name, output_csv_path)
    
    if df is not None:
        print("\nFirst few rows of the data:")
        print(df.head())
        
        print(f"\nDataFrame info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
    else:
        print("Failed to process the file.")

if __name__ == "__main__":
    main()
