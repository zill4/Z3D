#!/bin/bash

# Configuration
RABBITMQ_HOST=${RABBITMQ_HOST:-"localhost"}
RABBITMQ_PORT=${RABBITMQ_PORT:-"15672"}
RABBITMQ_USER=${RABBITMQ_USER:-"guest"}
RABBITMQ_PASS=${RABBITMQ_PASS:-"guest"}
QUEUE_NAME="image_processing"
TEST_IMAGE=${1:-"uploads/good.jpg"}

# Generate a unique job ID
JOB_ID=1234567890
echo "Generated Job ID: $JOB_ID"

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Copy test image to uploads directory with job ID as name
if [ -f "$TEST_IMAGE" ]; then
    cp "$TEST_IMAGE" "uploads/${JOB_ID}.png"
    echo "Copied test image to uploads/${JOB_ID}.png"
else
    echo "Error: Test image not found at $TEST_IMAGE"
    exit 1
fi

# Create JSON payload
JSON_PAYLOAD=$(cat <<EOF
{
    "properties": {
        "delivery_mode": 2
    },
    "routing_key": "$QUEUE_NAME",
    "payload": "{\"job_id\":\"$JOB_ID\",\"timestamp\":$(date +%s)}",
    "payload_encoding": "string"
}
EOF
)

# Send message to RabbitMQ using curl
echo "Sending message to RabbitMQ..."
curl -s -u "$RABBITMQ_USER:$RABBITMQ_PASS" -H "Content-Type: application/json" \
    -X POST -d "$JSON_PAYLOAD" \
    "http://$RABBITMQ_HOST:$RABBITMQ_PORT/api/exchanges/%2F/amq.default/publish"

echo -e "\nTest job submitted successfully!"
echo "Job ID: $JOB_ID"
echo -e "\nThe image processing service should now process this job."
echo "Check the service logs for progress updates."
echo -e "\nExpected output locations:"
echo "- Prepped image: prepped/${JOB_ID}_prepped.png"
echo "- 3D model: models/${JOB_ID}/textured.obj" 