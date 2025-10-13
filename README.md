# Creation Of a Large Language Model-Based Review Tool for Follow-Up Of Chest X-Rays

## Project Summary  
This project implements a comprehensive system for automating the analysis and grading of chest X-ray reports using Large Language Models (LLMs). The system combines Lunit AI analysis results with ground truth data to generate statistical reports, accuracy metrics, and identify discrepancies. It includes both a command-line interface and a web-based API service with an intuitive upload interface.

The goal is to classify reports into clinically relevant categories based on the urgency of follow-up required, ranging from normal (grade 1) to critical (grade 5), while providing comprehensive analysis tools for validation and quality assurance.
The core class grades from 1 to 5, though a binary grading system is used for the final production

Two complementary approaches were developed and evaluated:

1. **Semi-Algorithmic Approach**: Extracts structured findings from reports using an LLM, then applies rule-based grading based on finding types, temporal changes (new/stable/worsening), uncertainty levels, and medical device positioning.  

2. **All-LLM Approach**: Leverages an LLM to directly grade reports based on provided guidelines without the intermediate structured extraction step.

3. **Hybrid Method**: Combines semi-algorithmic results with LLM judgment ("second-think") to come to a reasoned grade

4. **Judge Method**: Evaluates the semialgo (1) and LLM (2) approaches to determine which is more appropriate as an ensemble system.

## Features

### Core Analysis Capabilities
- **Multiple Grading Methods**: Semi-algorithmic, pure LLM, hybrid, and judge-based approaches
- **Statistical Analysis**: Comprehensive accuracy metrics, agreement analysis, and time-based performance evaluation
- **False Negative Detection**: Automated identification and reporting of discrepancies between LLM and Lunit predictions
- **Multi-file Processing**: Support for combining multiple CARPL and RIS report files from different time periods

### Web Interface & API
- **Modern Web Interface**: Intuitive HTML interface for file uploads and result visualization
- **RESTful API**: FastAPI-based backend with comprehensive endpoints for analysis
- **Real-time Processing**: Asynchronous processing with status monitoring and progress tracking
- **Multiple Download Formats**: JSON, CSV, and specialized false negative reports

### Performance & Scalability
- **Concurrent Processing**: High-performance batch processing with configurable worker threads
- **Background Processing**: Non-blocking file analysis with task queuing
- **Flexible Model Support**: Ollama used for local LLM server through OpenAI-compatible API, ensuring privacy
- **Combined Server**: Single-file deployment option with both API and static file serving

## Repository Contents  
- scripts_audit/audit_cxr_v2.py: Python script implementing the grading approaches  
- scripts_audit/results_analysis.ipynb: Jupyter notebook containing analysis of the grading results, including statistical evaluations and visualizations  

## Implementation Details
Grading System (R)  
- R1: Normal without any findings  
- R2: Normal variant or minor pathology, does not require follow-up  
- R3: Abnormal, non-urgent follow-up required 
- R4: Abnormal, potentially important finding  
- R5: Critical, urgent follow-up required  
  
Data and Models  
- Dataset: 1114 anonymised primary care CXR reports
- Ground Truth: Expert radiologist annotation
- Models: Qwen3 32B, quantized to Q4_K_M, temperature=0.2, running on Ollama
- Evaluation Metrics: Exact agreement, Cohen's Kappa, sensitivity, specificity, F1 score, ROC-AUC

## Results
For the purposes of this project, the grades were further subdivided into 0: Normal (1-2), 1: Actionable (3, 4), 2: Critical (5).
All four approaches yielded good results, with a Cohen Kappa of up to 0.928 for the LLM approach and 0.846 for the Hybrid approach. 

If we further group the metrics into normal (0) and abnormal (everything else), the Cohen Kappa remains the same. The ROC-AUC of the LLM approach was 0.966 and the ROC-AUC of the Hybrid approach was 0.933.

McNemar's test revealed a significant difference between the accuracy of the Hybrid and the Judge approach (p=0.004).

## Technical Details

### Installation

#### Prerequisites
- Python 3.8+
- Access to OpenAI API or local Ollama installation

#### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/covie154/CXR-Audit
   ```

2. Install dependencies:
   ```bash
   pip install -r api_requirements.txt
   ```

3. Prepare configuration files:
   - `padchest_op.json`: Medical findings dictionary
   - `padchest_tubes_lines.json`: Tubes and lines dictionary  
   - `diagnoses.json`: Diagnoses dictionary

## Quick Start

### Web Interface (Recommended)

The easiest way to use the system is through the web interface:

1. **Start the combined server:**
   ```bash
   python combined_server.py
   ```

2. **Open your browser to:**
   ```
   http://localhost:1220
   ```

3. **Upload your files:**
   - CARPL results files (CSV/Excel)
   - RIS report files (CSV/Excel, password-protected supported)

4. **Get comprehensive results:**
   - Statistical analysis report
   - Accuracy metrics and agreement analysis
   - False negative case identification
   - Downloadable CSV data

### Alternative: Separate Servers

If you prefer to run API and static servers separately:

```bash
# Terminal 1: API Server (port 1221)
python api_server.py

# Terminal 2: Static Server (port 1220)  
python static_server.py
```

Or use the startup scripts:
```bash
# Windows
start_servers.bat

# PowerShell  
.\start_servers.ps1
```

## API Documentation

### API Endpoints

#### Single File Analysis
```http
POST /analyze
```
Upload single CARPL and RIS files for analysis.

**Request:**
- `lunit_file`: CARPL results file (CSV/Excel)
- `ground_truth_file`: RIS report file (CSV/Excel)  
- `priority_threshold`: Threshold for binarizing Lunit scores (default: 50.0)

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "queued",
  "message": "Files uploaded successfully. Processing started."
}
```

#### Multiple Files Analysis
```http
POST /analyze-multiple
```
Upload multiple CARPL and RIS files for combined analysis.

**Request:**
- `lunit_files`: Array of CARPL results files
- `ground_truth_files`: Array of RIS report files
- `priority_threshold`: Threshold for binarizing Lunit scores (default: 50.0)

#### Check Processing Status
```http
GET /status/{task_id}
```
Get current processing status and progress.

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "processing|completed|failed",
  "progress": "Current processing step...",
  "created_at": "2025-10-13T10:00:00Z",
  "completed_at": "2025-10-13T10:05:00Z"
}
```

#### Get Results
```http
GET /results/{task_id}
```
Retrieve completed analysis results.

**Response:**
```json
{
  "txt_report": "Statistical analysis text...",
  "data_output": [...],
  "csv_data": "Complete CSV data...",
  "filename": "cxr_analysis_2025-01-01_to_2025-01-31",
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

#### Command Line Usage

#### Basic Example
See batch_processing_eg.py for a basic example

```python
import pandas as pd
import json
from cxr_audit.grade_batch_async import BatchCXRProcessor

# Load configuration dictionaries
with open("padchest_op.json", "r") as f:
    padchest = json.load(f)
    
with open("padchest_tubes_lines.json", "r") as f:
    tubes_lines = json.load(f)

with open("diagnoses.json", "r") as f:
    diagnoses = json.load(f)

# Load your data
df = pd.read_csv("your_data.csv")

# Initialize batch processor
processor = BatchCXRProcessor(
    findings_dict=padchest,
    tubes_lines_dict=tubes_lines,
    diagnoses_dict=diagnoses,
    model_name="gpt-4o-mini",  # or "qwen3:32b-q4_K_M" for Ollama
    base_url="https://api.openai.com/v1",  # or "http://localhost:11434/v1" for Ollama
    api_key="your-api-key",  # or "dummy" for Ollama
    max_workers=4,
    rate_limit_delay=0.1
)

# Process the full pipeline
result_df = processor.process_full_pipeline(df, report_column='REPORT_TEXT')

# Save results
result_df.to_csv("graded_reports.csv", index=False)
```

#### Individual Method Usage

```python
# Process only specific methods
df_with_semialgo = processor.process_semialgo_batch(df, report_column='REPORT_TEXT')
df_with_llm = processor.process_llm_batch(df, report_column='REPORT_TEXT')
df_with_hybrid = processor.process_hybrid_batch(df, report_column='REPORT_TEXT')
df_with_judge = processor.process_judge_batch(df, report_column='REPORT_TEXT')
```

#### Single Report Processing

```python
# Initialize single classifier
classifier = CXRClassifier(
    findings=padchest,
    tubes_lines=tubes_lines,
    diagnoses=diagnoses,
    model_name="gpt-4o-mini"
)

# Grade single report
report_text = "Your chest X-ray report text here"
semialgo_result = classifier.gradeReportSemialgo(report_text)
llm_result = classifier.gradeReportLLM(report_text)
hybrid_result = classifier.gradeReportHybrid(report_text, semialgo_grade=3)
```

### Configuration

#### Model Configuration
- **OpenAI Models**: Any compatible model (e.g. `o4-mini`)
- **Ollama Models**: Any compatible model (e.g., `qwen3:32b-q4_K_M`, `llama3.1:8b`)

#### Performance Tuning
- `max_workers`: Number of concurrent processing threads (default: 5)
- `rate_limit_delay`: Delay between API calls in seconds (default: 0.1)

#### Data Format
Your input DataFrame should contain:
- A text column with the chest X-ray reports
- Optional ground truth columns for evaluation

### File Structure

```
scripts_lunit_review/
├── combined_server.py           # Single-file server (API + Static)
├── api_server.py               # API server (port 1221)
├── static_server.py            # Static file server (port 1220)
├── upload_interface.html       # Web interface
├── class_process_carpl.py      # Main analysis class
├── open_protected_xlsx.py      # Excel password handling
├── api_requirements.txt        # Python dependencies
├── start_servers.bat          # Windows startup script
├── start_servers.ps1          # PowerShell startup script
├── cxr_audit/
│   ├── lib_audit_cxr_v2.py    # Core CXR classifier
│   ├── grade_batch_async.py   # Batch processing
│   ├── helpers.py             # Utility functions
│   └── prompts.py             # LLM prompts
├── Readmes/
│   ├── MULTIPLE_FILES_GUIDE.md # Multiple file usage guide
│   └── example_*.py           # Example scripts
└── JSON configuration files:
    ├── padchest_op.json       # Medical findings
    ├── padchest_tubes_lines.json # Tubes and lines
    └── diagnoses.json         # Diagnoses dictionary
```

### Library Reference

#### ProcessCarpl

Main class for CARPL data analysis combining Lunit results with ground truth data.

##### Constructor
```python
ProcessCarpl(path_carpl_reports, path_ge_reports, processor=None, priority_threshold=10, passwd="GE_2024_P@55")
```
- `path_carpl_reports`: Single file path or list of CARPL files
- `path_ge_reports`: Single file path or list of RIS report files
- `priority_threshold`: Threshold for binarizing Lunit scores
- `passwd`: Password for protected Excel files

##### Key Methods
- `run_all()`: Complete analysis pipeline returning (processed_data, false_negatives_df, false_negatives_summary)
- `load_reports()`: Load and merge CARPL and RIS data
- `process_stats_accuracy()`: Calculate accuracy metrics and LLM comparisons
- `process_stats_time()`: Analyze time-based performance metrics
- `identify_false_negatives()`: Find cases where LLM disagrees with Lunit
- `txt_initial_metrics()`: Generate summary statistics text
- `txt_stats_accuracy()`: Generate accuracy analysis text
- `txt_stats_time()`: Generate time analysis text

#### BatchCXRProcessor

Concurrent batch processing of CXR reports for LLM grading.

##### Methods
- `process_full_pipeline(df, report_column)`: Run complete grading pipeline
- `process_semialgo_batch(df, report_column)`: Semi-algorithmic grading only
- `process_llm_batch(df, report_column)`: Pure LLM grading only
- `process_hybrid_batch(df, report_column)`: Hybrid grading only
- `process_judge_batch(df, report_column, manual_column=None)`: Judge-based comparison

#### CXRClassifier

Core classifier for individual report processing.

##### Methods
- `gradeReportSemialgo(report_text)`: Semi-algorithmic approach
- `gradeReportLLM(report_text)`: Pure LLM approach
- `gradeReportHybrid(report_text, semialgo_grade)`: Hybrid approach
- `gradeReportJudge(report_text, algo_grade, llm_grade, manual_grade=None)`: Judge approach

## Key Features

### False Negative Detection
The system automatically identifies cases where:
- LLM predicts negative (grade 0): Normal/no action needed
- Lunit predicts positive (grade 1): Abnormal/action needed

These discrepancies are highlighted in the web interface and available as separate CSV downloads for detailed review.

### Multiple File Support
Process multiple CARPL and RIS files from different time periods:
```python
# Multiple files example
processor = ProcessCarpl(
    path_carpl_reports=['carpl_jan.csv', 'carpl_feb.csv'],
    path_ge_reports=['ris_jan.xlsx', 'ris_feb.xlsx']
)
results = processor.run_all()
```

### Protected Excel Support
Automatically handles password-protected RIS Excel files:
- Default password: <REDACTED>
- Configurable via `passwd` parameter
- Supports both .xlsx and .xls formats

### Comprehensive Output
Each analysis provides:
- **Statistical Report**: Agreement metrics, Cohen's Kappa, sensitivity/specificity
- **Time Analysis**: Processing times, SLA compliance
- **False Negatives**: Detailed discrepancy analysis
- **CSV Export**: Complete processed data for further analysis

