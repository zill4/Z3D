import logging
import json
import os
import azure.functions as func
from azure.storage.blob import BlobServiceClient

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('GetJobStatus function processed a request.')
    
    # Get job_id from query parameter
    job_id = req.params.get('job_id')
    if not job_id:
        try:
            req_body = req.get_json()
            job_id = req_body.get('job_id')
        except ValueError:
            job_id = None
    
    if not job_id:
        return func.HttpResponse(
            "Please provide job_id parameter",
            status_code=400
        )
        
    # Connect to Azure Storage
    connection_string = os.environ["AzureWebJobsStorage"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Get status blob
    status_container = blob_service_client.get_container_client("status")
    status_blob = status_container.get_blob_client(f"{job_id}/status.json")
    
    try:
        # Get status data
        status_data = json.loads(status_blob.download_blob().readall())
        
        # If job is completed, add download URLs
        if status_data.get("status") == "completed":
            # Add download URLs for model files
            model_container = blob_service_client.get_container_client("models")
            prepped_container = blob_service_client.get_container_client("prepped")
            
            # Generate SAS tokens for access
            from datetime import datetime, timedelta
            from azure.storage.blob import generate_blob_sas, BlobSasPermissions
            
            # Add download URLs with SAS tokens
            status_data["download_urls"] = {}
            
            # Add prepped image URL
            prepped_blob_name = f"{job_id}_prepped.png"
            if "prepped_image" in status_data:
                prepped_blob_name = status_data["prepped_image"]
                
            prepped_sas = generate_blob_sas(
                account_name=blob_service_client.account_name,
                container_name=prepped_container.container_name,
                blob_name=prepped_blob_name,
                account_key=blob_service_client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=24)
            )
            
            status_data["download_urls"]["prepped_image"] = f"{prepped_container.url}/{prepped_blob_name}?{prepped_sas}"
            
            # Add model files URLs
            for blob_name in status_data.get("model_files", []):
                model_sas = generate_blob_sas(
                    account_name=blob_service_client.account_name,
                    container_name=model_container.container_name,
                    blob_name=blob_name,
                    account_key=blob_service_client.credential.account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=24)
                )
                
                file_name = blob_name.split("/")[-1]
                status_data["download_urls"][file_name] = f"{model_container.url}/{blob_name}?{model_sas}"
        
        return func.HttpResponse(
            json.dumps(status_data),
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Error getting status for job {job_id}: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "job_id": job_id,
                "status": "unknown",
                "error": f"Could not retrieve job status: {str(e)}"
            }),
            status_code=404,
            mimetype="application/json"
        ) 