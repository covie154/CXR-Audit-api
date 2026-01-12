#!/usr/bin/env python3
"""
Combined CXR Analysis Server
Runs both the API server (port 1221) and static file server (port 1220) in a single Python file.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
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
import uvicorn
import threading
import time

# Import your existing class
from class_process_carpl import ProcessCarpl
from open_protected_xlsx import open_protected_xlsx

thresholds = {
    'default': {
        'Atelectasis': 10,
        'Calcification': 10,
        'Cardiomegaly': 10,
        'Consolidation': 10,
        'Fibrosis': 10,
        'Mediastinal Widening': 10,
        'Nodule': 15,
        'Pleural Effusion': 10,
        'Pneumoperitoneum': 10,
        'Pneumothorax': 10,
        'Tuberculosis': 999
    },
    'YIS': {
        'Nodule': 5
    }
}

# ================================
# API SERVER (Port 1221)
# ================================

api_app = FastAPI(
    title="CXR Analysis API",
    description="API for processing chest X-ray reports using Lunit and ground truth data",
    version="1.0.0"
)

# Add CORS middleware for web frontend access
api_app.add_middleware(
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
            except Exception as e:
                try:
                    df = pd.read_excel(file_path)
                except Exception as e2:
                    print(f"Failed to open Excel file {file_path}: {str(e2)}")
                    raise HTTPException(status_code=400, detail=f"Error opening GE file {os.path.basename(file_path)}: {str(e2)}")
            finally:
                # Verify the Excel file has required columns
                if not {'ACCESSION_NO', 'TEXT_REPORT'}.issubset(df.columns):
                    missing_cols = {'ACCESSION_NO', 'TEXT_REPORT'} - set(df.columns)
                    raise HTTPException(status_code=400, detail=f"GE file {os.path.basename(file_path)} is missing required columns: {missing_cols}")
                ge_file_paths.append(file_path)
                print(f"Successfully identified GE file: {os.path.basename(file_path)}")
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


async def process_files_async(task_id: str, carpl_file_paths: List[str], ge_file_paths: List[str]):
    """Process files asynchronously in the background"""
    try:
        # Update status to processing
        processing_results[task_id]["status"] = "processing"
        processing_results[task_id]["progress"] = "Initializing ProcessCarpl..."
        
        # Initialize processor with multiple file paths
        processor = ProcessCarpl(carpl_file_paths, ge_file_paths, priority_threshold=thresholds)
        
        print("Starting task:", task_id)

        # Run the pipeline
        # For this API, we'll run the pipeline manually as we want to collate the text
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

@api_app.post("/analyze", response_model=ProcessingResponse)
async def analyze_files(
    background_tasks: BackgroundTasks,
    lunit_file: UploadFile = File(..., description="Lunit results file (CSV/Excel)"),
    ground_truth_file: UploadFile = File(..., description="Ground truth file (CSV/Excel)"),
    priority_threshold: float = 50.0
):
    """
    Upload Lunit and ground truth files for analysis.
    
    - **lunit_file**: CSV or Excel file containing Lunit analysis results
    - **ground_truth_file**: CSV or Excel file containing ground truth data
    - **priority_threshold**: Threshold for binarizing Lunit scores (0-100)
    """
    
    # Validate file types
    lunit_ext = os.path.splitext(lunit_file.filename or "")[1].lower()
    gt_ext = os.path.splitext(ground_truth_file.filename or "")[1].lower()
    
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    if lunit_ext not in allowed_extensions or gt_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Save uploaded files
        lunit_path = save_uploaded_file(lunit_file)
        gt_path = save_uploaded_file(ground_truth_file)
        
        # Initialize processing status
        processing_results[task_id] = {
            "status": "queued",
            "progress": "Files uploaded, queued for processing",
            "created_at": datetime.now().isoformat(),
            "lunit_filename": lunit_file.filename,
            "gt_filename": ground_truth_file.filename,
            "priority_threshold": priority_threshold
        }
        
        # Start background processing
        background_tasks.add_task(
            process_files_async, 
            task_id, 
            [lunit_path],  # Single file as list
            [gt_path]      # Single file as list
        )
        
        return ProcessingResponse(
            task_id=task_id,
            status="queued",
            message="Files uploaded successfully. Processing started."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app.post("/analyze-multiple", response_model=ProcessingResponse)
async def analyze_multiple_files(
    background_tasks: BackgroundTasks,
    lunit_files: List[UploadFile] = File(..., description="Multiple Lunit results files (CSV/Excel)"),
    ground_truth_files: List[UploadFile] = File(..., description="Multiple ground truth files (CSV/Excel)"),
    priority_threshold: float = 50.0
):
    """
    Upload multiple Lunit and ground truth files for combined analysis.
    
    - **lunit_files**: Multiple CSV or Excel files containing Lunit analysis results
    - **ground_truth_files**: Multiple CSV or Excel files containing ground truth data
    - **priority_threshold**: Threshold for binarizing Lunit scores (0-100)
    """
    
    # Validate all file types
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    
    for file in lunit_files + ground_truth_files:
        file_ext = os.path.splitext(file.filename or "")[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format for {file.filename}. Allowed: {', '.join(allowed_extensions)}"
            )
    
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Save all uploaded files
        lunit_paths = [save_uploaded_file(f) for f in lunit_files]
        gt_paths = [save_uploaded_file(f) for f in ground_truth_files]
        
        # Initialize processing status
        processing_results[task_id] = {
            "status": "queued",
            "progress": f"Files uploaded ({len(lunit_files)} Lunit, {len(ground_truth_files)} GT), queued for processing",
            "created_at": datetime.now().isoformat(),
            "lunit_filenames": [f.filename for f in lunit_files],
            "gt_filenames": [f.filename for f in ground_truth_files],
            "priority_threshold": priority_threshold
        }
        
        # Start background processing
        background_tasks.add_task(
            process_files_async, 
            task_id, 
            lunit_paths,
            gt_paths
        )
        
        return ProcessingResponse(
            task_id=task_id,
            status="queued",
            message=f"Multiple files uploaded successfully ({len(lunit_files)} Lunit, {len(ground_truth_files)} GT files). Processing started."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app.post("/analyze-auto-sort", response_model=ProcessingResponse)
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
            ge_file_paths
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

@api_app.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_processing_status(task_id: str):
    """Get the current status of a processing task."""
    
    if task_id not in processing_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = processing_results[task_id]
    return ProcessingStatus(
        task_id=task_id,
        status=result["status"],
        progress=result.get("progress"),
        results=result.get("results"),
        error=result.get("error"),
        created_at=result["created_at"],
        completed_at=result.get("completed_at")
    )

@api_app.get("/results/{task_id}")
async def get_results(task_id: str):
    """Get the results of a completed processing task."""
    
    if task_id not in processing_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = processing_results[task_id]
    
    if result["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task not completed. Current status: {result['status']}")
    
    return result["results"]

@api_app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a completed task and its results"""
    
    if task_id not in processing_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del processing_results[task_id]
    return {"message": "Task deleted successfully"}

@api_app.get("/tasks")
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

@api_app.get("/")
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
            "delete-task": "DELETE /tasks/{task_id} - Delete a task",
            "docs": "GET /docs - Interactive API documentation"
        }
    }

# ================================
# STATIC SERVER (Port 1220)
# ================================

static_app = FastAPI(
    title="CXR Analysis Static Server",
    description="Static file server for CXR Analysis upload interface",
    version="1.0.0"
)

# Add CORS middleware
static_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@static_app.get("/", response_class=HTMLResponse)
async def serve_upload_interface():
    """Serve the upload interface HTML page"""
    try:
        # Get the path to the HTML file in the same directory as this script
        html_path = os.path.join(os.path.dirname(__file__), "upload_interface.html")
        
        if not os.path.exists(html_path):
            raise HTTPException(status_code=404, detail="Upload interface not found")
        
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving HTML interface: {str(e)}")

@static_app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "message": "Static server is running"}

# ================================
# SERVER STARTUP FUNCTIONS
# ================================

def run_api_server():
    """Run the API server on port 1221"""
    print("Starting API Server on port 1221...")
    uvicorn.run(api_app, host="0.0.0.0", port=1221, log_level="info")

def run_static_server():
    """Run the static server on port 1220"""
    print("Starting Static Server on port 1220...")
    uvicorn.run(static_app, host="0.0.0.0", port=1220, log_level="info")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting CXR Analysis Combined Server")
    print("=" * 60)
    print("📊 API Server: http://localhost:1221")
    print("🌐 Upload Interface: http://localhost:1220")
    print("=" * 60)
    
    # Create threads for both servers
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    static_thread = threading.Thread(target=run_static_server, daemon=True)
    
    # Start both servers
    api_thread.start()
    time.sleep(1)  # Give API server a moment to start
    static_thread.start()
    
    print("✅ Both servers started successfully!")
    print("\n🌐 Open your browser to: http://localhost:1220")
    print("📋 API documentation: http://localhost:1221/docs")
    print("\n💡 Press Ctrl+C to stop both servers")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        print("✅ Goodbye!")