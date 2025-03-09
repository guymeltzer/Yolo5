FROM ultralytics/yolov5:latest-cpu

WORKDIR /usr/src/app

# Install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt

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
RUN apt-get update && apt-get install -y \
    unzip \
    curl \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip

# Expose port if necessary (e.g., for debugging)
# EXPOSE 5000

CMD ["python3", "app.py"]
