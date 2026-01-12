# LLM-Based Chest X-Ray Report Analysis Tool

## Overview
Automated analysis and grading of chest X-ray reports using Large Language Models (LLMs). Combines Lunit AI results with ground truth data to provide statistical reports, accuracy metrics, and discrepancy detection. Features both CLI and web-based API interface.

## Grading Approaches
This tool uses only two approaches from the original:
2. **All-LLM**: Direct LLM grading based on clinical guidelines
5. **Supplemental Analysis**: Finding the 10 Lunit findings from a report (WIP)

## Key Features
- **Statistical Analysis**: Accuracy metrics, Cohen's Kappa, ROC-AUC
- **Multi-file Support**: Combine CARPL and RIS files from multiple time periods
- **Web Interface**: Upload files, monitor progress, download results
- **RESTful API**: FastAPI backend with async processing
- **Concurrent Processing**: Configurable worker threads for batch processing
- **Privacy-Focused**: Local LLM via Ollama/OpenAI-compatible API

## Grading System
- ***R1**: Normal without findings
- **R2**: Normal variant, no follow-up needed
- **R3**: Abnormal, non-urgent follow-up
- **R4**: Abnormal, potentially important
- **R5**: Critical, urgent follow-up*

Binary classification: 0 (Normal: R1-2), 1 (Actionable: R3-4), 2 (Critical: R5)

## Results
- **Dataset**: 1114 anonymized primary care CXR reports
- **LLM Approach**: Cohen's κ=0.928, ROC-AUC=0.966
- **Hybrid Approach**: Cohen's κ=0.846, ROC-AUC=0.933
- **Model**: Qwen3 32B (Q4_K_M quantization, temp=0.2) via Ollama

## Installation

```bash
git clone https://github.com/covie154/CXR-Audit-api
cd CXR-Audit-api
pip install -r api_requirements.txt
```

**Required Configuration Files:**
- `padchest_op.json` - Medical findings dictionary
- `padchest_tubes_lines.json` - Tubes and lines dictionary
- `diagnoses.json` - Diagnoses dictionary

## Quick Start

### Web Interface (Recommended)
```bash
python combined_server.py
```
Open browser to `http://localhost:1220`

**Upload:** CARPL files (CSV/Excel) + RIS reports (CSV/Excel, password-protected supported)  
**Download:** Statistical reports, accuracy metrics, false negative cases (JSON/CSV)

### Separate Servers
```bash
# API Server (port 1221)
python api_server.py

# Static Server (port 1220)
python static_server.py
```

## API Usage

### Single File Analysis
```python
import requests

with open('carpl.csv', 'rb') as f1, open('ris.xlsx', 'rb') as f2:
    response = requests.post('http://localhost:1221/analyze',
        files={'lunit_file': f1, 'ground_truth_file': f2},
        data={'priority_threshold': 50.0})
    task_id = response.json()['task_id']

# Check status
status = requests.get(f'http://localhost:1221/status/{task_id}').json()

# Get results
results = requests.get(f'http://localhost:1221/results/{task_id}').json()
```

### Multiple Files Analysis
```python
response = requests.post('http://localhost:1221/analyze-multiple',
    files=[
        ('lunit_files', open('carpl1.csv', 'rb')),
        ('lunit_files', open('carpl2.csv', 'rb')),
        ('ground_truth_files', open('ris1.xlsx', 'rb')),
        ('ground_truth_files', open('ris2.xlsx', 'rb'))
    ],
    data={'priority_threshold': 50.0})
```

### Main Endpoints
- `POST /analyze` - Single file analysis
- `POST /analyze-multiple` - Multiple file analysis
- `GET /status/{task_id}` - Check processing status
- `GET /results/{task_id}` - Retrieve results
- `GET /download-csv/{task_id}` - Download CSV
- `GET /download-false-negatives/{task_id}` - Download false negatives
  "false_negatives": {
    "summary": {
      "count": 5,
      "total_cases": 100,
      "percentage": 5.0,
      "description": "Cases where LLM predicted negative but Lunit predicted positive"
    },
    "data": [...],
    "csv_data": "False negatives CSV data..."
  }
}
```

### Usage

#### Web Interface Usage

1. **Select Upload Mode:**
   - Single Files: One CARPL + one RIS file
   - Multiple Files: Combine files from different time periods

2. **Upload Files:**
   - Drag & drop or click to select
   - Remove files with "✕ Remove" button
   - Password-protected Excel files supported

3. **Monitor Progress:**
   - Real-time status updates
   - Progress bar with current step

4. **View Results:**
   - Statistical analysis report
   - False negative cases with site and report details
   - Download options: JSON, CSV, False Negatives CSV

#### Programmatic API Usage

```python
import requests
import time

# Upload files for analysis
with open('carpl_file.csv', 'rb') as f1, open('ris_file.xlsx', 'rb') as f2:
    response = requests.post(
        'http://localhost:1221/analyze',
        files={
            'lunit_file': f1,
            'ground_truth_file': f2
        },
        data={'priority_threshold': 50.0}
    )

task_id = response.json()['task_id']

# Monitor progress
while True:
    status_response = requests.get(f'http://localhost:1221/status/{task_id}')
    status = status_response.json()
    
    if status['status'] == 'completed':
        break
    elif status['status'] == 'failed':
        print("Processing failed:", status.get('error'))
        break
    
    print(f"Progress: {status.get('progress', 'Processing...')}")
    time.sleep(2)

# Get results
results = requests.get(f'http://localhost:1221/results/{task_id}').json()
print("Analysis complete!")
print(f"False negatives found: {results['false_negatives']['summary']['count']}")
```

## Configuration

### Model Support
- Anything! Including 'gpt-5' (OpenAI), 'qwen3:32b-q4_K_M' (Ollama)

### Performance Tuning
- `max_workers`: Concurrent threads (default: 5)
- `rate_limit_delay`: API delay in seconds (default: 0.1)
- `priority_threshold`: Lunit score threshold (default: 50.0)

## Repository Structure
```
CXR-Audit-api/
├── combined_server.py          # Combined API + static server
├── api_server.py              # API server (port 1221)
├── static_server.py           # Static server (port 1220)
├── upload_interface.html      # Web UI
├── class_process_carpl.py     # CARPL/RIS analysis
├── open_protected_xlsx.py     # Password-protected Excel handler
├── api_requirements.txt       # Dependencies
├── cxr_audit/                 # Core grading library
│   ├── lib_audit_cxr_v2.py   # CXR classifier
│   ├── grade_batch_async.py  # Batch processor
│   ├── helpers.py            # Utilities
│   └── prompts.py            # LLM prompts
├── Readmes/                   # Documentation & examples
└── *.json                     # Medical dictionaries
```

## Key Classes

### `ProcessCarpl`
Main analysis class combining Lunit CARPL and RIS reports.

**Constructor:**
```python
ProcessCarpl(path_carpl_reports, path_ge_reports, 
             processor=None, supplemental_steps=True,
             priority_threshold=50.0, passwd="GE_2024_P@55")
```

**Methods:**
- `run_all()` - Complete pipeline → (data, false_negatives_df, summary)
- `load_reports()` - Load and merge CARPL/RIS data
- `process_stats_accuracy()` - Calculate metrics
- `identify_false_negatives()` - Find LLM/Lunit discrepancies

### `BatchCXRProcessor`
Concurrent batch processing for CXR reports.

**Methods:**
- `process_full_pipeline(df, report_column)` - All grading methods
- `process_semialgo_batch(df, report_column)` - Semi-algorithmic only
- `process_llm_batch(df, report_column)` - LLM only
- `process_hybrid_batch(df, report_column)` - Hybrid only
- `process_judge_batch(df, report_column)` - Judge comparison

### `CXRClassifier`
Individual report classification.

**Methods:**
- `gradeReportSemialgo(report_text)` - Semi-algorithmic grading
- `gradeReportLLM(report_text)` - LLM grading
- `gradeReportHybrid(report_text, semialgo_grade)` - Hybrid grading
- `gradeReportJudge(report_text, algo_grade, llm_grade)` - Judge grading

## Features Detail

### False Negative Detection
Identifies cases where LLM predicts normal (grade 0) but Lunit predicts abnormal (grade 1+). Available as separate CSV download with site and report details.

### Multiple File Support
```python
# Combine files from different time periods
processor = ProcessCarpl(
    path_carpl_reports=['jan.csv', 'feb.csv', 'mar.csv'],
    path_ge_reports=['jan.xlsx', 'feb.xlsx', 'mar.xlsx']
)
```

### Password-Protected Excel
Automatically handles encrypted RIS files. (configurable via `passwd` parameter).
