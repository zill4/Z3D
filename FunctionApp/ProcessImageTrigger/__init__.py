import logging
import json
import os
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import (
    ContainerGroup, Container, ResourceRequirements, 
    ResourceRequests, EnvironmentVariable, GpuResource
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
    
    # Initialize status
    status_container = blob_service_client.get_container_client("status")
    try:
        status_container.create_container()
    except:
        pass
        
    # Upload image to blob storage
    uploads_container = blob_service_client.get_container_client("uploads")
    try:
        uploads_container.create_container()
    except:
        pass
        
    # Upload image from URL to blob storage
    from urllib.request import urlopen
    image_data = urlopen(image_url).read()
    uploads_container.upload_blob(f"{job_id}.png", image_data, overwrite=True)
    
    # Initialize status
    status_blob = status_container.get_blob_client(f"{job_id}/status.json")
    status_data = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    status_blob.upload_blob(json.dumps(status_data), overwrite=True)
    
    # Check if there's an available container or create one
    queue_container = blob_service_client.get_container_client("queue")
    try:
        queue_container.create_container()
    except:
        pass
    
    # Add job to queue
    queue_blob = queue_container.get_blob_client("job_queue.json")
    try:
        queue_data = json.loads(queue_blob.download_blob().readall())
        queue_data["jobs"].append(job_id)
    except:
        queue_data = {"jobs": [job_id]}
    
    queue_blob.upload_blob(json.dumps(queue_data), overwrite=True)
    
    # Check if processor container exists
    container_client = ContainerInstanceManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["SUBSCRIPTION_ID"]
    )
    
    container_group_name = "z3d-processor"
    resource_group = os.environ["RESOURCE_GROUP"]
    
    try:
        container_group = container_client.container_groups.get(
            resource_group, container_group_name
        )
        
        # Container exists - check its state
        if container_group.instance_view.state == "Running":
            # Container is already running, job will be picked up from queue
            logging.info(f"Container {container_group_name} is already running")
        else:
            # Container exists but is stopped - start it
            logging.info(f"Container {container_group_name} exists but is stopped - starting it")
            container_client.container_groups.begin_restart(
                resource_group, container_group_name
            )
            
        return func.HttpResponse(
            json.dumps({
                "job_id": job_id,
                "status": "queued",
                "message": "Job added to queue. Container will process it.",
                "status_url": f"https://{os.environ['FUNCTION_APP_NAME']}.azurewebsites.net/api/GetJobStatus?job_id={job_id}"
            }),
            mimetype="application/json"
        )
            
    except Exception as e:
        logging.info(f"Container {container_group_name} doesn't exist - creating it")
        # Container doesn't exist, create it
        # Create the ACI with GPU
        container = Container(
            name="image-processor",
            image=os.environ["CONTAINER_IMAGE"],
            resources=ResourceRequirements(
                requests=ResourceRequests(
                    memory_in_gb=32.0,  # More memory for model loading
                    cpu=8.0,            # More CPU cores
                    gpu=GpuResource(count=1, sku="V100")  # Upgraded GPU
                )
            ),
            environment_variables=[
                EnvironmentVariable(name="AZURE_STORAGE_CONNECTION_STRING", 
                                   value=connection_string),
                EnvironmentVariable(name="UPLOADS_CONTAINER", value="uploads"),
                EnvironmentVariable(name="PREPPED_CONTAINER", value="prepped"),
                EnvironmentVariable(name="MODELS_CONTAINER", value="models"),
                EnvironmentVariable(name="STATUS_CONTAINER", value="status"),
                EnvironmentVariable(name="QUEUE_CONTAINER", value="queue"),
                EnvironmentVariable(name="IDLE_TIMEOUT_MINUTES", value="30")  # Keep container for 30 mins
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
                "status": "queued",
                "message": "Job queued. Starting processor.",
                "status_url": f"https://{os.environ['FUNCTION_APP_NAME']}.azurewebsites.net/api/GetJobStatus?job_id={job_id}"
            }),
            mimetype="application/json"
        ) 