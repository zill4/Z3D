import os
import json
import uuid
import logging
import time
import threading
from pathlib import Path
import traceback
from datetime import datetime, timedelta

# Azure packages
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

# Image processing
import torch
import numpy as np
from PIL import Image
from rembg import remove, new_session

# Hunyuan 3D packages
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hunyuan3d-service")

# Configuration
MAX_FACE_COUNT = 2000  # Target for optimization
MODEL_CACHE_DIR = Path('models/hunyuan')
IDLE_TIMEOUT_MINUTES = int(os.getenv("IDLE_TIMEOUT_MINUTES", "30"))

# Initialize background removal session
bg_remover_session = new_session()

# Azure Storage client
blob_service_client = BlobServiceClient.from_connection_string(
    os.getenv("AZURE_STORAGE_CONNECTION_STRING")
)
uploads_container = os.getenv("UPLOADS_CONTAINER", "uploads")
prepped_container = os.getenv("PREPPED_CONTAINER", "prepped")
models_container = os.getenv("MODELS_CONTAINER", "models")
status_container = os.getenv("STATUS_CONTAINER", "status")
queue_container = os.getenv("QUEUE_CONTAINER", "queue")

# Global pipelines (load once, reuse for all jobs)
shape_pipeline = None
paint_pipeline = None
last_job_time = datetime.now()
shutdown_timer = None

def ensure_containers():
    """Ensure all necessary blob containers exist"""
    for container_name in [uploads_container, prepped_container, models_container, status_container, queue_container]:
        try:
            blob_service_client.create_container(container_name)
            logger.info(f"Created container: {container_name}")
        except ResourceExistsError:
            logger.info(f"Container already exists: {container_name}")

def update_job_status(job_id, status, progress=None, message=None, error=None):
    """Update job status in Azure Blob Storage and trigger callback"""
    status_blob_name = f"{job_id}/status.json"
    status_data = {
        "job_id": job_id,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "progress": progress,
        "message": message,
        "error": error
    }
    
    # Upload status to blob storage
    status_container_client = blob_service_client.get_container_client(status_container)
    status_container_client.upload_blob(
        name=status_blob_name,
        data=json.dumps(status_data),
        overwrite=True
    )
    
    logger.info(f"Updated job {job_id} status to {status} - Progress: {progress}%")
    
    # Send webhook callback to backend
    callback_url = os.getenv("CALLBACK_URL")
    if callback_url:
        try:
            import requests
            requests.post(
                callback_url,
                json={
                    "job_id": job_id,
                    "status": status,
                    "progress": progress,
                    "message": message,
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat()
                },
                timeout=5
            )
            logger.info(f"Sent callback for job {job_id}")
        except Exception as e:
            logger.error(f"Callback failed: {str(e)}")
    
    return status_data

def download_image(job_id):
    """Download image from Azure Blob Storage"""
    update_job_status(job_id, "downloading", 5, "Downloading image")
    
    blob_name = f"{job_id}.png"
    blob_client = blob_service_client.get_blob_client(container=uploads_container, blob=blob_name)
    
    # Create local directories
    os.makedirs("uploads", exist_ok=True)
    local_path = f"uploads/{job_id}.png"
    
    # Download the blob
    with open(local_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    
    logger.info(f"Downloaded image for job {job_id}")
    return local_path

def remove_background(job_id, input_path):
    """Remove image background and save the result"""
    update_job_status(job_id, "processing", 15, "Removing background")
    logger.info(f"Removing background from {input_path}")
    
    # Read the input image
    with open(input_path, "rb") as input_file:
        input_content = input_file.read()
    
    # Configure rembg to be less aggressive with artwork
    output_content = remove(
        input_content,
        session=bg_remover_session,
        bgcolor=(255, 255, 255, 0),
        alpha_matting=True,
        alpha_matting_foreground_threshold=180,  # Even less aggressive
        alpha_matting_background_threshold=20,
        alpha_matting_erode_size=5
    )
    
    # Generate output path
    prepped_path = f"prepped/{job_id}_prepped.png"
    os.makedirs("prepped", exist_ok=True)
    
    with open(prepped_path, "wb") as output_file:
        output_file.write(output_content)
    
    # Upload to Azure
    prepped_blob_client = blob_service_client.get_blob_client(
        container=prepped_container, 
        blob=f"{job_id}_prepped.png"
    )
    
    with open(prepped_path, "rb") as data:
        prepped_blob_client.upload_blob(data, overwrite=True)
    
    logger.info(f"Background removed and uploaded for job {job_id}")
    update_job_status(job_id, "processing", 30, "Background removed")
    
    return prepped_path

def load_pipelines():
    """Initialize and load Hunyuan3D pipelines"""
    global shape_pipeline, paint_pipeline
    
    if shape_pipeline is not None and paint_pipeline is not None:
        logger.info("Reusing existing pipelines")
        return shape_pipeline, paint_pipeline
    
    logger.info("Loading Hunyuan3D pipelines")
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize shape pipeline
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        cache_dir=str(MODEL_CACHE_DIR / 'shape_pipeline')
    )
    
    # Initialize texture pipeline
    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    
    # Configure pipeline parameters
    paint_pipeline.view_size = 512      # Smaller view size for faster processing
    paint_pipeline.texture_size = 1024  # Smaller texture for faster processing
    
    return shape_pipeline, paint_pipeline

def generate_3d_model(job_id, image_path):
    """Generate 3D model from image using Hunyuan3D"""
    global last_job_time
    
    logger.info(f"Generating 3D model from {image_path}")
    update_job_status(job_id, "processing", 40, "Initializing 3D generation")
    
    # Create output directory
    output_dir = f"models/{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize pipelines
        shape_pipeline, paint_pipeline = load_pipelines()
        
        # Generate shape
        update_job_status(job_id, "processing", 50, "Generating 3D shape")
        mesh = shape_pipeline(image=image_path)[0]
        
        # Log original face count
        logger.info(f"Original mesh faces: {len(mesh.faces)}")
        update_job_status(job_id, "processing", 60, "Optimizing mesh")
        
        # Process mesh - remove artifacts and degenerate faces
        processed_mesh = FloaterRemover()(mesh)
        processed_mesh = DegenerateFaceRemover()(processed_mesh)
        
        # Reduce face count aggressively for speed
        processed_mesh = FaceReducer()(processed_mesh, max_facenum=MAX_FACE_COUNT)
        
        # Additional quality checks
        processed_mesh = processed_mesh.process(validate=True)
        processed_mesh.fill_holes()
        
        logger.info(f"Processed mesh faces: {len(processed_mesh.faces)}")
        update_job_status(job_id, "processing", 70, "Generating texture")
        
        # Save base mesh
        base_mesh_path = os.path.join(output_dir, "base_mesh.obj")
        processed_mesh.export(
            base_mesh_path,
            include_normals=True,
            include_texture=True
        )
        
        # Upload base mesh to Azure
        upload_model_files(job_id, output_dir, "base_mesh.obj")
        
        # Generate texture
        update_job_status(job_id, "processing", 80, "Applying textures")
        textured_mesh = paint_pipeline(processed_mesh, image=image_path)
        
        # Save final textured mesh
        textured_path = os.path.join(output_dir, "textured.obj")
        material_path = os.path.join(output_dir, "textured.mtl")
        texture_path = os.path.join(output_dir, "textured.png")
        
        textured_mesh.export(
            textured_path,
            include_normals=True,
            include_texture=True,
            resolver=None,
            mtl_name='textured.mtl'
        )
        
        # Upload all model files to Azure
        update_job_status(job_id, "processing", 90, "Uploading model files")
        model_files = upload_model_files(job_id, output_dir, "textured.obj")
        
        update_job_status(
            job_id, 
            "completed", 
            100, 
            "3D model generation completed",
            {
                "model_files": model_files,
                "prepped_image": f"{job_id}_prepped.png"
            }
        )
        
        # Update last job time
        last_job_time = datetime.now()
        
        logger.info(f"Completed 3D model generation for job {job_id}")
        return textured_path
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"3D generation failed for job {job_id}: {error_msg}")
        logger.error(traceback.format_exc())
        
        update_job_status(job_id, "error", None, "3D generation failed", error_msg)
        
        # Try to save whatever mesh we have as fallback
        try:
            fallback_path = os.path.join(output_dir, "untextured_mesh.glb")
            mesh.export(fallback_path)
            upload_model_files(job_id, output_dir, "untextured_mesh.glb")
            logger.info(f"Saved fallback mesh for job {job_id}")
            return fallback_path
        except:
            logger.error(f"Failed to save fallback mesh for job {job_id}")
            return None

def upload_model_files(job_id, output_dir, main_file):
    """Upload all model files to Azure Blob Storage"""
    models_container_client = blob_service_client.get_container_client(models_container)
    uploaded_files = []
    
    # Upload the main file and its dependencies
    for file_path in Path(output_dir).glob("*"):
        if file_path.is_file():
            blob_name = f"{job_id}/{file_path.name}"
            with open(file_path, "rb") as data:
                models_container_client.upload_blob(blob_name, data, overwrite=True)
            
            # Generate SAS URL for access
            uploaded_files.append(blob_name)
            logger.info(f"Uploaded {blob_name} to Azure Storage")
    
    return uploaded_files

def process_job(job_id):
    """Process a single job with the given ID"""
    global last_job_time
    
    try:
        # Update status to started
        update_job_status(job_id, "started", 0, "Job started")
        
        # Step 1: Download the image
        input_path = download_image(job_id)
        
        # Step 2: Remove background
        prepped_path = remove_background(job_id, input_path)
        
        # Step 3: Generate 3D model
        model_path = generate_3d_model(job_id, prepped_path)
        
        # Update last job time
        last_job_time = datetime.now()
        
        return {
            "status": "success",
            "job_id": job_id,
            "model_path": model_path,
            "prepped_path": prepped_path
        }
    
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        update_job_status(job_id, "error", None, "Job failed", str(e))
        return {
            "status": "error",
            "job_id": job_id,
            "error": str(e)
        }

def check_queue():
    """Check for new jobs in the queue"""
    global shutdown_timer
    
    # Reset shutdown timer
    if shutdown_timer:
        shutdown_timer.cancel()
    
    # Get the queue blob
    queue_container_client = blob_service_client.get_container_client(queue_container)
    queue_blob = queue_container_client.get_blob_client("job_queue.json")
    
    try:
        # Get queued jobs
        queue_data = json.loads(queue_blob.download_blob().readall())
        jobs = queue_data.get("jobs", [])
        
        if jobs:
            # Get the next job
            job_id = jobs.pop(0)
            
            # Update the queue
            queue_container_client.upload_blob(
                name="job_queue.json",
                data=json.dumps({"jobs": jobs}),
                overwrite=True
            )
            
            # Process the job
            logger.info(f"Processing job {job_id} from queue")
            process_job(job_id)
            
            # Check for more jobs
            check_queue()
        else:
            # No jobs in queue, start shutdown timer
            logger.info(f"No jobs in queue. Will check again in 30 seconds.")
            time_since_last_job = (datetime.now() - last_job_time).total_seconds() / 60
            
            if time_since_last_job > IDLE_TIMEOUT_MINUTES:
                logger.info(f"No activity for {IDLE_TIMEOUT_MINUTES} minutes. Shutting down container.")
                return
            
            # Check again in 30 seconds
            shutdown_timer = threading.Timer(30, check_queue)
            shutdown_timer.daemon = True
            shutdown_timer.start()
    
    except Exception as e:
        logger.error(f"Error checking queue: {str(e)}")
        
        # Check again in 30 seconds
        shutdown_timer = threading.Timer(30, check_queue)
        shutdown_timer.daemon = True
        shutdown_timer.start()

def main():
    """Main entry point for the service"""
    global last_job_time
    
    logger.info("Starting Hunyuan3D Image Processing Service for Azure")
    
    # Ensure Azure containers exist
    ensure_containers()
    
    # Initialize last job time
    last_job_time = datetime.now()
    
    # Load pipelines (one time)
    load_pipelines()
    
    # Start queue checker
    check_queue()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(60)
            
            # Check if we should shut down
            time_since_last_job = (datetime.now() - last_job_time).total_seconds() / 60
            if time_since_last_job > IDLE_TIMEOUT_MINUTES:
                logger.info(f"No activity for {IDLE_TIMEOUT_MINUTES} minutes. Shutting down container.")
                break
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    
    return 0

if __name__ == "__main__":
    exit(main())

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "gpu": torch.cuda.is_available(),
        "memory_usage": psutil.virtual_memory().percent,
        "uptime": time.time() - start_time
    })