import pika
import json
import os
from rembg import remove
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import traceback
import logging
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from rembg import new_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hunyuan3d-service")

# Initialize background removal session
bg_remover_session = new_session()

# Configuration
MAX_FACE_COUNT = 2000  # Target for optimization
MODEL_CACHE_DIR = Path('models/hunyuan')

def ensure_dirs():
    """Ensure all necessary directories exist"""
    for dir_path in ['uploads', 'prepped', 'models', MODEL_CACHE_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def remove_background(input_path):
    """Remove image background and save the result"""
    logger.info(f"Removing background from {input_path}")
    with open(input_path, "rb") as input_file:
        input_content = input_file.read()
    
    # Remove background with white background
    output_content = remove(input_content, session=bg_remover_session, bgcolor=(255, 255, 255, 0))
    
    prepped_path = input_path.replace(".png", "_prepped.png")
    with open(prepped_path, "wb") as output_file:
        output_file.write(output_content)
    
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

def process_job(job_id):
    """Process a single job with the given ID"""
    try:
        # Define paths
        input_path = f"uploads/{job_id}.png"
        output_dir = f"models/{job_id}"
        
        # Step 1: Remove background
        prepped_path = remove_background(input_path)
        
        # Step 2: Generate 3D model
        model_path = generate_3d_model(prepped_path, output_dir)
        
        # Return the result info
        return {
            "status": "success",
            "job_id": job_id,
            "model_path": model_path,
            "prepped_path": prepped_path
        }
    
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "job_id": job_id,
            "error": str(e)
        }

def callback(ch, method, properties, body):
    """RabbitMQ message callback handler"""
    job = json.loads(body.decode())
    job_id = job["job_id"]
    
    logger.info(f"Received job {job_id}")
    
    # Process the job
    result = process_job(job_id)
    
    # Log the result
    if result["status"] == "success":
        logger.info(f"Successfully processed job {job_id}")
    else:
        logger.error(f"Failed to process job {job_id}: {result['error']}")
    
    # Acknowledge the message
    ch.basic_ack(delivery_tag=method.delivery_tag)
    
    # If needed, send a response to a result queue
    if "reply_to" in job:
        response_queue = job["reply_to"]
        ch.basic_publish(
            exchange='',
            routing_key=response_queue,
            body=json.dumps(result)
        )

def main():
    """Main entry point for the service"""
    logger.info("Starting Hunyuan3D Image Processing Service")
    
    # Ensure directories exist
    ensure_dirs()
    
    # RabbitMQ setup
    rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
    rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
    rabbitmq_pass = os.getenv("RABBITMQ_PASS", "guest")
    
    # Connection parameters with credentials
    credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_pass)
    parameters = pika.ConnectionParameters(
        host=rabbitmq_host,
        credentials=credentials,
        heartbeat=600,  # Longer heartbeat for longer processing tasks
        blocked_connection_timeout=300
    )
    
    # Retry connection with backoff
    retry_count = 0
    max_retries = 5
    while retry_count < max_retries:
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.queue_declare(queue="image_processing", durable=True)
            
            # Set prefetch to 1 to ensure we only process one job at a time
            channel.basic_qos(prefetch_count=1)
            
            # Register the callback
            channel.basic_consume(queue="image_processing", on_message_callback=callback)
            
            logger.info("Service started. Waiting for jobs...")
            channel.start_consuming()
            break
            
        except pika.exceptions.AMQPConnectionError as e:
            retry_count += 1
            wait_time = 2 ** retry_count  # Exponential backoff
            logger.error(f"Connection failed (attempt {retry_count}/{max_retries}). Retrying in {wait_time}s: {str(e)}")
            import time
            time.sleep(wait_time)
    
    if retry_count >= max_retries:
        logger.critical("Failed to connect to RabbitMQ after maximum retries")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())