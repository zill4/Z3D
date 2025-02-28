import logging
import json
import os
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import (
    ContainerGroup, Container, ContainerGroupNetworkProtocol,
    ResourceRequirements, ResourceRequests, EnvironmentVariable,
    ContainerGroupRestartPolicy, GpuResource
)
from azure.identity import DefaultAzureCredential

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get job ID from request
    try:
        req_body = req.get_json()
        job_id = req_body.get('job_id')
        image_url = req_body.get('image_url')
    except ValueError:
        return func.HttpResponse(
            "Please pass a job_id and image_url in the request body",
            status_code=400
        )

    if not job_id or not image_url:
        return func.HttpResponse(
            "Please provide both job_id and image_url values",
            status_code=400
        )

    # Connect to Azure Storage
    connection_string = os.environ["AzureWebJobsStorage"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Create status container if it doesn't exist
    status_container = blob_service_client.get_container_client("status")
    try:
        status_container.create_container()
    except:
        # Container already exists
        pass
    
    # Initialize status
    status_blob = status_container.get_blob_client(f"{job_id}/status.json")
    status_data = {
        "job_id": job_id,
        "status": "initializing",
        "progress": 0,
        "message": "Starting job",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    status_blob.upload_blob(json.dumps(status_data), overwrite=True)
    
    # Download image from URL and upload to blob storage
    uploads_container = blob_service_client.get_container_client("uploads")
    try:
        uploads_container.create_container()
    except:
        # Container already exists
        pass
    
    # Create the ACI with GPU
    container_client = ContainerInstanceManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["SUBSCRIPTION_ID"]
    )
    
    # Start the container instance with our image
    container_group_name = f"z3d-job-{job_id}"
    resource_group = os.environ["RESOURCE_GROUP"]
    
    # Container configuration
    container = Container(
        name="image-processor",
        image=os.environ["CONTAINER_IMAGE"],
        resources=ResourceRequirements(
            requests=ResourceRequests(
                memory_in_gb=8.0,
                cpu=2.0,
                gpu=GpuResource(count=1, sku="K80")
            )
        ),
        environment_variables=[
            EnvironmentVariable(name="JOB_ID", value=job_id),
            EnvironmentVariable(name="AZURE_STORAGE_CONNECTION_STRING", 
                               value=connection_string),
            EnvironmentVariable(name="UPLOADS_CONTAINER", value="uploads"),
            EnvironmentVariable(name="PREPPED_CONTAINER", value="prepped"),
            EnvironmentVariable(name="MODELS_CONTAINER", value="models"),
            EnvironmentVariable(name="STATUS_CONTAINER", value="status")
        ]
    )
    
    # Create container group
    container_group = ContainerGroup(
        location=os.environ["LOCATION"],
        containers=[container],
        os_type="Linux",
        restart_policy="Never"
    )
    
    # Deploy the container
    container_client.container_groups.begin_create_or_update(
        resource_group,
        container_group_name,
        container_group
    )
    
    return func.HttpResponse(
        json.dumps({
            "job_id": job_id,
            "status": "started",
            "message": "Processing started. Monitor status at container level.",
            "status_url": f"https://{os.environ['FUNCTION_APP_NAME']}.azurewebsites.net/api/GetJobStatus?job_id={job_id}"
        }),
        mimetype="application/json"
    ) 