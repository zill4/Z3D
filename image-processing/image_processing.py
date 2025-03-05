import json
import os
from rembg import remove
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import traceback
import logging
from azure.storage.queue import QueueClient
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableClient
from azure.identity import DefaultAzureCredential
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from rembg import new_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hunyuan3d-service")

# Configuration
MAX_FACE_COUNT = 2000
MODEL_CACHE_DIR = Path('models/hunyuan')

# Azure Storage settings
STORAGE_ACCOUNT = os.getenv("STORAGE_ACCOUNT_NAME", "z3dstorage")
QUEUE_NAME = "processing-queue"

# Container names
CONTAINERS = {
    'uploads': 'images',
    'prepped': 'prepped',
    'models': 'models'
}

# Initialize Azure clients with managed identity
credential = DefaultAzureCredential()

def get_blob_service_client():
    return BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=credential
    )

def get_table_client():
    return TableClient(
        endpoint=f"https://{STORAGE_ACCOUNT}.table.core.windows.net",
        table_name="jobstatus",
        credential=credential
    )

def get_queue_client():
    return QueueClient(
        account_url=f"https://{STORAGE_ACCOUNT}.queue.core.windows.net",
        queue_name=QUEUE_NAME,
        credential=credential
    )

def update_job_status(job_id, status, error=None):
    """Update job status in Table Storage"""
    table_client = get_table_client()
    entity = {
        'PartitionKey': 'jobs',
        'RowKey': job_id,
        'status': status
    }
    if error:
        entity['error'] = error
    table_client.update_entity(entity)

def download_blob(container_name, blob_name, local_path):
    """Download blob from Azure Storage"""
    blob_service_client = get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as file:
        data = blob_client.download_blob()
        file.write(data.readall())

def upload_blob(container_name, blob_name, local_path):
    """Upload blob to Azure Storage"""
    blob_service_client = get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    with open(local_path, "rb") as file:
        blob_client.upload_blob(file, overwrite=True)

def remove_background(input_path):
    """Remove image background and save the result"""
    logger.info(f"Removing background from {input_path}")
    
    # Read the input image
    with open(input_path, "rb") as input_file:
        input_content = input_file.read()
    
    # Configure rembg to be less aggressive
    output_content = remove(
        input_content,
        session=bg_remover_session,
        bgcolor=(255, 255, 255, 0),
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )
    
    # Generate output path properly preserving the job ID
    input_path = Path(input_path)
    prepped_path = input_path.parent / f"{input_path.stem}_prepped{input_path.suffix}"
    
    with open(prepped_path, "wb") as output_file:
        output_file.write(output_content)
    
    return str(prepped_path)

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
    
    # Configure pipeline parameters for faster processing and lower quality
    paint_pipeline.view_size = 512    # Smaller view size for faster processing
    paint_pipeline.texture_size = 1024  # Smaller texture for faster processing
    
    return shape_pipeline, paint_pipeline

def generate_3d_model(image_path, output_dir):
    """Generate 3D model from image using Hunyuan3D"""
    logger.info(f"Generating 3D model from {image_path}")
    
    # Initialize pipelines
    shape_pipeline, paint_pipeline = load_pipelines()
    
    try:
        # Generate shape
        logger.info("Generating 3D shape...")
        mesh = shape_pipeline(image=image_path)[0]
        
        # Log original face count
        logger.info(f"Original mesh faces: {len(mesh.faces)}")
        
        # Process mesh - remove artifacts and degenerate faces
        processed_mesh = FloaterRemover()(mesh)
        processed_mesh = DegenerateFaceRemover()(processed_mesh)
        
        # Reduce face count aggressively for speed
        processed_mesh = FaceReducer()(processed_mesh, max_facenum=MAX_FACE_COUNT)
        
        # Additional quality checks
        processed_mesh = processed_mesh.process(validate=True)
        processed_mesh.fill_holes()
        
        logger.info(f"Processed mesh faces: {len(processed_mesh.faces)}")
        
        # Generate texture
        logger.info("Generating texture...")
        
        # Create output directory for this job
        os.makedirs(output_dir, exist_ok=True)
        
        # Save base mesh
        base_mesh_path = os.path.join(output_dir, "base_mesh.obj")
        processed_mesh.export(
            base_mesh_path,
            include_normals=True,
            include_texture=True
        )
        
        # Generate texture
        textured_mesh = paint_pipeline(processed_mesh, image=image_path)
        
        # Save final textured mesh
        textured_path = os.path.join(output_dir, "textured.obj")
        textured_mesh.export(
            textured_path,
            include_normals=True,
            include_texture=True,
            resolver=None,
            mtl_name='textured.mtl'
        )
        
        logger.info(f"Completed 3D model generation, saved at {textured_path}")
        return textured_path
        
    except Exception as e:
        logger.error(f"3D generation failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try to save whatever mesh we have as fallback
        try:
            fallback_path = os.path.join(output_dir, "untextured_mesh.glb")
            mesh.export(fallback_path)
            logger.info(f"Saved fallback mesh at {fallback_path}")
            return fallback_path
        except:
            logger.error("Failed to save fallback mesh")
            return None

def process_job(job_data):
    """Process a single job"""
    try:
        job_id = job_data["job_id"]
        image_url = job_data["image_url"]
        
        # Extract blob path from URL
        blob_path = image_url.split(f"{CONTAINERS['uploads']}/")[1]
        local_input_path = f"local_storage/{CONTAINERS['uploads']}/{blob_path}"
        
        # Download input image
        download_blob(CONTAINERS['uploads'], blob_path, local_input_path)
        
        # Update status
        update_job_status(job_id, "processing")
        
        # Process image (your existing processing code)
        prepped_path = remove_background(local_input_path)
        model_path = generate_3d_model(prepped_path, f"local_storage/{CONTAINERS['models']}/{job_id}")
        
        # Upload results
        upload_blob(CONTAINERS['models'], f"{job_id}/model.obj", model_path)
        
        # Update final status
        update_job_status(job_id, "completed")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        update_job_status(job_id, "failed", str(e))

def process_single_message():
    """Process a single message from the queue and exit"""
    try:
        # Get a single message
        queue_client = get_queue_client()
        messages = queue_client.receive_messages(max_messages=1)
        
        for msg in messages:
            try:
                # Process message
                job_data = json.loads(msg.content)
                process_job(job_data)
                
                # Delete message after processing
                queue_client.delete_message(msg)
                return True
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                logger.error(traceback.format_exc())
                return False
        
        return False  # No messages to process
        
    except Exception as e:
        logger.error(f"Queue processing error: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point for the service"""
    logger.info("Starting Hunyuan3D Image Processing Service")
    
    # Create local storage directories
    os.makedirs("local_storage", exist_ok=True)
    for container in CONTAINERS.values():
        os.makedirs(f"local_storage/{container}", exist_ok=True)
    
    # Initialize background removal session
    global bg_remover_session
    bg_remover_session = new_session()
    
    # Process single message and exit
    success = process_single_message()
    
    # Exit with appropriate status code
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())