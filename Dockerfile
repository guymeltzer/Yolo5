# Use Ubuntu 24.04 as the base image
FROM ultralytics/yolov5:v6.2-cpu

# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    unzip \
    curl

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Download YOLO model
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt

# Ensure the 'data' directory exists before downloading coco128.yaml
RUN mkdir -p data && \
    curl -L https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco128.yaml -o data/coco128.yaml

# Copy application files
COPY . .

# Define environment variables
ENV S3_BUCKET_NAME=${S3_BUCKET_NAME}
ENV SQS_QUEUE_URL=${SQS_QUEUE_URL}
ENV MONGO_URI="mongodb://mongo1:27017,mongo2:27017,mongo3:27017"
ENV MONGO_DB="default_db"
ENV MONGO_COLLECTION="predictions"
ENV POLYBOT_URL="http://polybot:5000/results"

# Install AWS CLI (if needed for any AWS operations)
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    curl \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws
