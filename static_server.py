#!/usr/bin/env python3
"""
Static file server to serve the upload interface on port 1220.
The API server continues to run on port 1221.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

# Create a simple FastAPI app for static file serving
app = FastAPI(
    title="CXR Analysis Static Server",
    description="Static file server for CXR Analysis upload interface",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
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

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "message": "Static server is running"}

if __name__ == "__main__":
    print("Starting static file server on http://localhost:1220")
    print("Upload interface will be available at: http://localhost:1220")
    print("Make sure the API server is running on port 1221")
    uvicorn.run("static_server:app", host="0.0.0.0", port=1220, reload=True)