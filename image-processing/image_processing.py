import os
import json
import uuid
import logging
import time
from pathlib import Path
import traceback
from datetime import datetime

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

def ensure_containers():
    """Ensure all necessary blob containers exist"""
    for container_name in [uploads_container, prepped_container, models_container, status_container]:
        try:
            blob_service_client.create_container(container_name)
            logger.info(f"Created container: {container_name}")
        except ResourceExistsError:
            logger.info(f"Container already exists: {container_name}")

def update_job_status(job_id, status, progress=None, message=None, error=None):
    """Update job status in Azure Blob Storage"""
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
        alpha_matting_foreground_threshold=200,  # Less aggressive (was 240)
        alpha_matting_background_threshold=20,   # More inclusive (was 10)
        alpha_matting_erode_size=5               # Less erosion (was 10)
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
    try:
        # Update status to started
        update_job_status(job_id, "started", 0, "Job started")
        
        # Step 1: Download the image
        input_path = download_image(job_id)
        
        # Step 2: Remove background
        prepped_path = remove_background(job_id, input_path)
        
        # Step 3: Generate 3D model
        model_path = generate_3d_model(job_id, prepped_path)
        
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

def main():
    """Main entry point for the service"""
    logger.info("Starting Hunyuan3D Image Processing Service for Azure")
    
    # Ensure Azure containers exist
    ensure_containers()
    
    # Get job ID from environment or command line
    job_id = os.getenv("JOB_ID")
    
    if job_id:
        logger.info(f"Processing job {job_id}")
        result = process_job(job_id)
        logger.info(f"Job {job_id} completed with status: {result['status']}")
    else:
        logger.error("No JOB_ID provided in environment variables")
    
    return 0

if __name__ == "__main__":
    exit(main())