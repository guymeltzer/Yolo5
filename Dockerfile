FROM ultralytics/yolov5:latest-cpu
WORKDIR /usr/src/app

# Install dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt

# Copy application files
COPY . .

# Set environment variables
ENV BUCKET_NAME=my-s3-bucket
ENV SQS_QUEUE_URL=https://sqs.eu-north-1.amazonaws.com/123456789012/my-queue
ENV MONGO_URI=mongodb://mongo1:27017,mongo2:27017,mongo3:27017
ENV MONGO_DB=default_db
ENV MONGO_COLLECTION=predictions
ENV POLYBOT_URL=http://polybot:5000/results

CMD ["python3", "app.py"]
