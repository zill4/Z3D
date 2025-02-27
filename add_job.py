import pika
import json
import os

rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
channel = connection.channel()
channel.queue_declare(queue="image_processing")

job = {
    "job_id": "test_image",
    # Normally includes blob_url, but here we assume the file is in uploads/
}
channel.basic_publish(
    exchange="",
    routing_key="image_processing",
    body=json.dumps(job)
)

print("Added job to queue")
connection.close()