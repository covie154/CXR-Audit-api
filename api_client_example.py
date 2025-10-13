import requests
import time
import json

# API client example
class CXRAnalysisClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def analyze_files(self, lunit_file_path, gt_file_path, priority_threshold=50.0):
        """Upload files for analysis"""
        url = f"{self.base_url}/analyze"
        
        with open(lunit_file_path, 'rb') as lunit_file, \
             open(gt_file_path, 'rb') as gt_file:
            
            files = {
                'lunit_file': (lunit_file_path, lunit_file),
                'ground_truth_file': (gt_file_path, gt_file)
            }
            
            data = {'priority_threshold': priority_threshold}
            
            response = requests.post(url, files=files, data=data)
            
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def get_status(self, task_id):
        """Check processing status"""
        url = f"{self.base_url}/status/{task_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def get_results(self, task_id):
        """Get analysis results"""
        url = f"{self.base_url}/results/{task_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 202:
            return {"status": "processing", "message": "Not yet complete"}
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def wait_for_completion(self, task_id, max_wait_time=300, check_interval=5):
        """Wait for processing to complete"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = self.get_status(task_id)
            
            if status['status'] == 'completed':
                return self.get_results(task_id)
            elif status['status'] == 'failed':
                raise Exception(f"Processing failed: {status.get('error', 'Unknown error')}")
            
            print(f"Status: {status['status']} - {status.get('progress', 'Processing...')}")
            time.sleep(check_interval)
        
        raise Exception("Processing timed out")

# Example usage
if __name__ == "__main__":
    client = CXRAnalysisClient()
    
    # Upload files
    try:
        response = client.analyze_files(
            lunit_file_path="path/to/lunit_file.csv",
            gt_file_path="path/to/ground_truth_file.csv",
            priority_threshold=50.0
        )
        
        task_id = response['task_id']
        print(f"Analysis started. Task ID: {task_id}")
        
        # Wait for completion
        results = client.wait_for_completion(task_id)
        
        # Print results
        print("\n=== ANALYSIS RESULTS ===")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
