# ProcessCarpl Multiple Files Support

## Overview

The `ProcessCarpl` class has been updated to support multiple input files for both CARPL (deployment statistics) and GE (RIS reports) data sources. This enhancement allows you to combine data from multiple time periods or sources in a single analysis.

## Key Features

### ✅ Backward Compatibility
- Single file inputs still work exactly as before
- No changes needed to existing code

### 🔄 Multiple File Support
- Accept lists of files for both CARPL and GE inputs
- Automatically combines data from all files
- Supports mixed file formats (CSV and Excel)

### 🚀 API Integration
- New `/analyze-multiple` endpoint for multiple files
- Existing `/analyze` endpoint unchanged

## Usage Examples

### Single Files (Backward Compatible)
```python
from class_process_carpl import ProcessCarpl

# This still works exactly as before
processor = ProcessCarpl(
    path_carpl_reports="deployment_stats14.csv",
    path_ge_reports="RIS_WeeklyReport.xlsx"
)
```

### Multiple Files
```python
# Multiple CARPL files
carpl_files = [
    "deployment_stats_week1.csv",
    "deployment_stats_week2.csv",
    "deployment_stats_week3.csv"
]

# Multiple GE files
ge_files = [
    "RIS_WeeklyReport_week1.xlsx",
    "RIS_WeeklyReport_week2.xlsx", 
    "RIS_WeeklyReport_week3.xlsx"
]

processor = ProcessCarpl(
    path_carpl_reports=carpl_files,
    path_ge_reports=ge_files
)
```

### Mixed Single and Multiple
```python
# Single CARPL file, multiple GE files
processor = ProcessCarpl(
    path_carpl_reports="deployment_stats.csv",
    path_ge_reports=[
        "RIS_week1.xlsx",
        "RIS_week2.xlsx"
    ]
)
```

## File Format Support

### CARPL Files
- ✅ CSV files (`.csv`)
- ✅ Excel files (`.xlsx`, `.xls`)

### GE Files  
- ✅ CSV files (`.csv`)
- ✅ Excel files (`.xlsx`, `.xls`) - supports password protection
- ✅ Mixed formats in same analysis

## API Usage

### Single Files Endpoint
```http
POST /analyze
Content-Type: multipart/form-data

lunit_file: deployment_stats.csv
ground_truth_file: ris_report.xlsx
priority_threshold: 50.0
```

### Multiple Files Endpoint
```http
POST /analyze-multiple
Content-Type: multipart/form-data

lunit_files: [deployment_stats1.csv, deployment_stats2.csv]
ground_truth_files: [ris_report1.xlsx, ris_report2.xlsx]
priority_threshold: 50.0
```

## Implementation Details

### Data Combination Process

1. **CARPL Files**:
   - Each file is loaded using `pd.read_csv()` or `pd.read_excel()`
   - All dataframes are concatenated using `pd.concat()`
   - Duplicate accession numbers are preserved (handled during merge)

2. **GE Files**:
   - CSV files loaded with `pd.read_csv()`
   - Excel files loaded with `open_protected_xlsx()` for password support
   - All dataframes are concatenated using `pd.concat()`

3. **Merging**:
   - Combined datasets are merged on accession number
   - Inner join ensures only matching records are included
   - Same merge logic as single-file version

### Load Progress Output
```
Loading CARPL file: deployment_stats_week1.csv
  -> Loaded 1250 records from deployment_stats_week1.csv
Loading CARPL file: deployment_stats_week2.csv  
  -> Loaded 1180 records from deployment_stats_week2.csv
Combined CARPL files: 2430 total records

Loading GE file: RIS_WeeklyReport_week1.xlsx
  -> Loaded 1300 records from RIS_WeeklyReport_week1.xlsx
Loading GE file: RIS_WeeklyReport_week2.xlsx
  -> Loaded 1200 records from RIS_WeeklyReport_week2.xlsx  
Combined GE files: 2500 total records

Total loaded: 2430 CARPL records and 2500 GE records. Merged to 2200 entries.
```

## Error Handling

### File Validation
- Checks file extensions for supported formats
- Validates file existence before processing
- Handles password-protected Excel files

### Graceful Degradation
- If a file fails to load, the error is reported
- Processing continues with successfully loaded files
- Clear error messages indicate which file caused issues

## Performance Considerations

### Memory Usage
- All files loaded into memory simultaneously
- Consider system RAM when processing large file sets
- Monitor memory usage for very large datasets

### Processing Time
- Linear increase with number of files
- Dominated by file I/O operations
- Parallel processing not implemented (future enhancement)

## Migration Guide

### From Single Files
No changes required! Your existing code will continue to work.

### To Multiple Files
1. Update file paths from strings to lists:
   ```python
   # Before
   processor = ProcessCarpl("file1.csv", "file2.xlsx")
   
   # After  
   processor = ProcessCarpl(["file1.csv", "file2.csv"], ["file2.xlsx", "file3.xlsx"])
   ```

2. Update API calls to use `/analyze-multiple` endpoint

## Testing

Run the test suite to verify functionality:
```bash
python test_multiple_files.py
```

View examples:
```bash
python example_multiple_files.py
```

## Future Enhancements

- [ ] Parallel file loading for improved performance
- [ ] Streaming processing for very large datasets  
- [ ] Data validation across combined files
- [ ] Automatic file discovery in directories
- [ ] Progress callbacks for long operations