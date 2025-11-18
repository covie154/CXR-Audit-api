from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
import tempfile
import os
import uuid
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import asyncio
from datetime import datetime

# Import your existing class
from class_process_carpl import ProcessCarpl
from open_protected_xlsx import open_protected_xlsx

app = FastAPI(
    title="CXR Analysis API",
    description="API for processing chest X-ray reports using Lunit and ground truth data",
    version="1.0.0"
)

# Add CORS middleware for web frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for processing results (use Redis/database in production)
processing_results = {}

# Response models
class ProcessingResponse(BaseModel):
    task_id: str
    status: str
    message: str

class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

class AnalysisResults(BaseModel):
    txt_report: str
    data_output: Any  # This can be more specifically typed if needed
    csv_data: str
    filename: str

def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location and return path"""
    try:
        # Create temporary file with proper extension
        if not upload_file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename")
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = upload_file.file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        upload_file.file.seek(0)  # Reset file pointer
        return tmp_file_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error saving file: {str(e)}")
    
async def sort_files_async(file_paths: List[str]) -> Dict[str, List[str]]:
    """Sort uploaded files into GE (Excel) and CARPL (CSV) categories"""
    ge_file_paths = []
    carpl_file_paths = []
    
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.xlsx', '.xls']:
            # Attempt to open with protected Excel handler
            try:
                df = open_protected_xlsx(file_path, password="GE_2024_P@55")
                # Verify the Excel file has required columns
                if not {'ACCESSION_NO', 'TEXT_REPORT'}.issubset(df.columns):
                    missing_cols = {'ACCESSION_NO', 'TEXT_REPORT'} - set(df.columns)
                    raise HTTPException(status_code=400, detail=f"GE file {os.path.basename(file_path)} is missing required columns: {missing_cols}")
                ge_file_paths.append(file_path)
                print(f"Successfully identified GE file: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Failed to open Excel file {file_path}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error opening GE file {os.path.basename(file_path)}: {str(e)}")
        elif ext == '.csv':
            try:
                df = pd.read_csv(file_path)
                # Verify the CSV file has required columns
                if not {'ACCESSION_NO', "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Fibrosis", "Mediastinal Widening", \
                            "Nodule", "Pleural Effusion", "Pneumoperitoneum", "Pneumothorax", "Tuberculosis"}.issubset(df.columns):
                    missing_cols = {'ACCESSION_NO', "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Fibrosis", "Mediastinal Widening", \
                                    "Nodule", "Pleural Effusion", "Pneumoperitoneum", "Pneumothorax", "Tuberculosis"} - set(df.columns)
                    raise HTTPException(status_code=400, detail=f"CARPL file {os.path.basename(file_path)} is missing required columns: {missing_cols}")
                carpl_file_paths.append(file_path)
                print(f"Successfully identified CARPL file: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Failed to open CSV file {file_path}: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error opening CARPL file {os.path.basename(file_path)}: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {os.path.basename(file_path)}")

    return {
        "ge_file_paths": ge_file_paths,
        "carpl_file_paths": carpl_file_paths
    }


async def process_files_async(task_id: str, carpl_file_paths: List[str], ge_file_paths: List[str], 
                             priority_threshold: float = 50.0):
    """Async function to process files in background - supports multiple files"""
    try:
        # Update status to processing
        processing_results[task_id]["status"] = "processing"
        processing_results[task_id]["progress"] = f"Initializing ProcessCarpl with {len(carpl_file_paths)} CARPL files and {len(ge_file_paths)} GE files..."
        
        # Initialize processor with multiple file paths
        processor = ProcessCarpl(carpl_file_paths, ge_file_paths)
        
        print(f"Starting task {task_id} with {len(carpl_file_paths)} CARPL files and {len(ge_file_paths)} GE files")

        # Run the pipeline
        # For this API, we'll run the pipeline manually as we want to collate the text
        # It runs around 2 reports/second on my machine (2.2-2.5 it/s)
        df_merged = processor.load_reports()
        total_seconds = len(df_merged) / 2  # 2 reports per second
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        processing_results[task_id]["progress"] = f"Loaded {len(df_merged)} reports. This may take around {minutes} minutes and {seconds} seconds to complete."
        initial_metrics = processor.txt_initial_metrics(df_merged)
        
        df_merged = processor.process_stats_accuracy(df_merged)
        processing_results[task_id]["progress"] = "Calculated accuracy statistics"
        stats_accuracy = processor.txt_stats_accuracy(df_merged)

        df_merged = processor.process_stats_time(df_merged)
        processing_results[task_id]["progress"] = "Calculated time statistics"
        stats_time = processor.txt_stats_time(df_merged)

        # We disable this for now because it messes with the workflow
        #processor.box_time(df_merged)
        
        # Identify false negatives before rearranging columns
        false_negative_df, false_negative_summary = processor.identify_false_negatives(df_merged)
        processing_results[task_id]["progress"] = "Identified false negative cases"
        
        # Finally we rearrange the columns to fit the proper format
        df_merged = processor.rearrange_columns(df_merged)
        print("Final output has a shape of:", df_merged.shape)
        df_merged_json = df_merged.to_json(orient='records', indent=4)
        
        # Convert DataFrame to CSV string
        df_merged_csv = df_merged.to_csv(index=False)
        
        # Convert false negative DataFrame to CSV and JSON
        false_negative_csv = false_negative_df.to_csv(index=False) if not false_negative_df.empty else ""
        false_negative_json = false_negative_df.to_json(orient='records', indent=4) if not false_negative_df.empty else "[]"
        
        first_date = df_merged['PROCEDURE_END_DATE'].min()
        last_date = df_merged['PROCEDURE_END_DATE'].max()

        output_json = {
            "txt_report": initial_metrics + "\n" + stats_accuracy + "\n" + stats_time,
            "data_output": json.loads(df_merged_json),
            "filename": f"cxr_analysis_{first_date}_to_{last_date}",
            "csv_data": df_merged_csv,
            "false_negatives": {
                "summary": false_negative_summary,
                "data": json.loads(false_negative_json),
                "csv_data": false_negative_csv
            }
        }

        # Update final status
        processing_results[task_id]["status"] = "completed"
        processing_results[task_id]["results"] = output_json
        processing_results[task_id]["completed_at"] = datetime.now().isoformat()
        processing_results[task_id]["progress"] = "Analysis complete"
        
        print("Processing complete for task:", task_id)
        
        # Clean up temporary files
        try:
            for file_path in carpl_file_paths + ge_file_paths:
                os.unlink(file_path)
        except:
            pass  # Files might already be deleted
            
    except Exception as e:
        processing_results[task_id]["status"] = "failed"
        processing_results[task_id]["error"] = str(e)
        processing_results[task_id]["completed_at"] = datetime.now().isoformat()

@app.post("/analyze", response_model=ProcessingResponse)
async def analyze_files(
    background_tasks: BackgroundTasks,
    lunit_file: UploadFile = File(..., description="Lunit results file (CSV/Excel)"),
    ground_truth_file: UploadFile = File(..., description="Ground truth file (CSV/Excel)"),
    priority_threshold: float = 50.0
):
    """
    Upload Lunit and ground truth files for analysis.
    
    - **lunit_file**: CSV or Excel file containing Lunit analysis results
    - **ground_truth_file**: CSV or Excel file containing ground truth labels
    - **priority_threshold**: Threshold for binarizing Lunit scores (default: 50.0)
    """
    
    # Validate file types
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    
    if not lunit_file.filename:
        raise HTTPException(status_code=400, detail="Lunit file must have a filename")
    if not ground_truth_file.filename:
        raise HTTPException(status_code=400, detail="Ground truth file must have a filename")
    
    lunit_ext = os.path.splitext(lunit_file.filename)[1].lower()
    gt_ext = os.path.splitext(ground_truth_file.filename)[1].lower()
    
    if lunit_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Lunit file must be CSV or Excel format")
    
    if gt_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Ground truth file must be CSV or Excel format")
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded files
    try:
        lunit_file_path = save_uploaded_file(lunit_file)
        gt_file_path = save_uploaded_file(ground_truth_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing uploaded files: {str(e)}")
    
    # Initialize task status
    processing_results[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "progress": "Files uploaded successfully",
        "results": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "files": {
            "lunit_filename": lunit_file.filename,
            "gt_filename": ground_truth_file.filename
        }
    }
    
    # Start background processing
    background_tasks.add_task(
        process_files_async, 
        task_id, 
        [lunit_file_path],  # Convert single file to list
        [gt_file_path],     # Convert single file to list
        priority_threshold
    )
    
    return ProcessingResponse(
        task_id=task_id,
        status="queued",
        message="Files uploaded successfully. Processing started in background."
    )

@app.post("/analyze-multiple", response_model=ProcessingResponse)
async def analyze_multiple_files(
    background_tasks: BackgroundTasks,
    lunit_files: List[UploadFile] = File(..., description="Multiple Lunit results files (CSV/Excel)"),
    ground_truth_files: List[UploadFile] = File(..., description="Multiple ground truth files (CSV/Excel)"),
    priority_threshold: float = 50.0
):
    """
    Upload multiple Lunit and ground truth files for analysis.
    
    - **lunit_files**: List of CSV or Excel files containing Lunit analysis results
    - **ground_truth_files**: List of CSV or Excel files containing ground truth labels
    - **priority_threshold**: Threshold for binarizing Lunit scores (default: 50.0)
    """
    
    # Validate file types
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    
    # Validate all lunit files
    for lunit_file in lunit_files:
        if not lunit_file.filename:
            raise HTTPException(status_code=400, detail="Lunit file must have a filename")
        lunit_ext = os.path.splitext(lunit_file.filename)[1].lower()
        if lunit_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Lunit file {lunit_file.filename} must be CSV or Excel format")
    
    # Validate all ground truth files
    for gt_file in ground_truth_files:
        if not gt_file.filename:
            raise HTTPException(status_code=400, detail="Ground truth file must have a filename")
        gt_ext = os.path.splitext(gt_file.filename)[1].lower()
        if gt_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Ground truth file {gt_file.filename} must be CSV or Excel format")
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded files
    try:
        lunit_file_paths = []
        for lunit_file in lunit_files:
            lunit_file_path = save_uploaded_file(lunit_file)
            lunit_file_paths.append(lunit_file_path)
        
        gt_file_paths = []
        for gt_file in ground_truth_files:
            gt_file_path = save_uploaded_file(gt_file)
            gt_file_paths.append(gt_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing uploaded files: {str(e)}")
    
    # Initialize task status
    processing_results[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "progress": "Files uploaded successfully",
        "results": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "files": {
            "lunit_filenames": [f.filename for f in lunit_files],
            "gt_filenames": [f.filename for f in ground_truth_files]
        }
    }
    
    # Start background processing
    background_tasks.add_task(
        process_files_async, 
        task_id, 
        lunit_file_paths,
        gt_file_paths,
        priority_threshold
    )
    
    return ProcessingResponse(
        task_id=task_id,
        status="queued",
        message=f"Multiple files uploaded successfully ({len(lunit_files)} Lunit, {len(ground_truth_files)} GE). Processing started in background."
    )

@app.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_processing_status(task_id: str):
    """Get the status of a processing task"""
    
    if task_id not in processing_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return ProcessingStatus(**processing_results[task_id])

@app.get("/results/{task_id}")
async def get_results(task_id: str):
    """Get the complete results of a processing task"""
    
    if task_id not in processing_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = processing_results[task_id]
    
    if task["status"] == "failed":
        raise HTTPException(status_code=400, detail=f"Processing failed: {task['error']}")
    
    if task["status"] != "completed":
        raise HTTPException(status_code=202, detail="Processing not yet complete")
    
    return JSONResponse(content=task["results"])

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a completed task and its results"""
    
    if task_id not in processing_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del processing_results[task_id]
    return {"message": "Task deleted successfully"}

@app.post("/analyze-auto-sort", response_model=ProcessingResponse)
async def analyze_auto_sort_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Mixed files (will be auto-sorted into Lunit/GE categories)"),
    priority_threshold: float = 50.0
):
    """
    Upload mixed files and automatically sort them into Lunit (CSV) and GE (Excel) categories for analysis.
    
    - **files**: List of mixed CSV or Excel files (will be automatically categorized)
    - **priority_threshold**: Threshold for binarizing Lunit scores (default: 50.0)
    """
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded files
    try:
        all_file_paths = []
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="All files must have filenames")
            file_path = save_uploaded_file(file)
            all_file_paths.append(file_path)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing uploaded files: {str(e)}")
    
    # Initialize task status
    processing_results[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "progress": "Files uploaded successfully, sorting files...",
        "results": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "files": {
            "uploaded_filenames": [f.filename for f in files]
        }
    }
    
    # Sort files and start background processing
    try:
        file_sorting = await sort_files_async(all_file_paths)
        carpl_file_paths = file_sorting["carpl_file_paths"]
        ge_file_paths = file_sorting["ge_file_paths"]
        
        # Update task status with sorting results
        processing_results[task_id]["progress"] = f"Files sorted: {len(carpl_file_paths)} CARPL (CSV), {len(ge_file_paths)} GE (Excel)"
        processing_results[task_id]["files"]["sorted_files"] = {
            "carpl_count": len(carpl_file_paths),
            "ge_count": len(ge_file_paths)
        }
        
        if not carpl_file_paths:
            raise HTTPException(status_code=400, detail="No CARPL (CSV) files found in uploaded files")
        if not ge_file_paths:
            raise HTTPException(status_code=400, detail="No GE (Excel) files found in uploaded files")
        
        # Start background processing
        background_tasks.add_task(
            process_files_async, 
            task_id, 
            carpl_file_paths,
            ge_file_paths,
            priority_threshold
        )
        
    except Exception as e:
        processing_results[task_id]["status"] = "failed"
        processing_results[task_id]["error"] = str(e)
        processing_results[task_id]["completed_at"] = datetime.now().isoformat()
        # Clean up files on error
        try:
            for file_path in all_file_paths:
                os.unlink(file_path)
        except:
            pass
        raise HTTPException(status_code=400, detail=str(e))
    
    return ProcessingResponse(
        task_id=task_id,
        status="queued",
        message=f"Files uploaded and sorted successfully ({len(carpl_file_paths)} CARPL, {len(ge_file_paths)} GE). Processing started in background."
    )

@app.get("/tasks")
async def list_tasks():
    """List all tasks with their current status"""
    
    task_list = []
    for task_id, task_data in processing_results.items():
        task_list.append({
            "task_id": task_id,
            "status": task_data["status"],
            "created_at": task_data["created_at"],
            "completed_at": task_data.get("completed_at"),
            "files": task_data.get("files", {})
        })
    
    return {"tasks": task_list}

@app.get("/")
async def root():
    """API health check and information"""
    return {
        "message": "CXR Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze - Upload single files for analysis",
            "analyze-multiple": "POST /analyze-multiple - Upload multiple files for analysis",
            "analyze-auto-sort": "POST /analyze-auto-sort - Upload mixed files (auto-sorted into categories)",
            "status": "GET /status/{task_id} - Check processing status",
            "results": "GET /results/{task_id} - Get analysis results",
            "tasks": "GET /tasks - List all tasks",
            "docs": "GET /docs - Interactive API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=1221, reload=True)
