import time
import ultralytics
import os
import json
import boto3
import requests
import uuid
import yaml
from pathlib import Path
from loguru import logger
from pymongo import MongoClient

# --- Environment Variables ---
images_bucket = os.environ['S3_BUCKET_NAME']
queue_url = os.environ['SQS_QUEUE_URL']
mongo_uri = os.environ['MONGO_URI']
db_name = os.environ.get('MONGO_DB', 'default_db')
collection_name = os.environ.get('MONGO_COLLECTION', 'predictions')
polybot_url = os.environ.get('POLYBOT_URL', 'http://polybot-service:30619/results')

# --- AWS Clients ---
sqs_client = boto3.client('sqs', region_name='eu-north-1')
s3_client = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])

# --- MongoDB Client ---
try:
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client[db_name]
    collection = db[collection_name]
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    # Optionally, retry or exit with a specific status code
    exit(1)

# --- Load Class Names ---
try:
    with open("data/coco128.yaml", "r") as stream:
        names = yaml.safe_load(stream)['names']
except Exception as e:
    logger.error(f"Failed to load class names: {e}")
    exit(1)

def consume():
    """Polls SQS queue for messages and processes image jobs."""
    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=5
            )

            if 'Messages' in response:
                message = response['Messages'][0]
                receipt_handle = message['ReceiptHandle']

                # Parse job details
                job = json.loads(message['Body'])
                img_name = job.get('img_name')
                chat_id = job.get('chat_id')
                prediction_id = str(uuid.uuid4())

                if not img_name or not chat_id:
                    logger.error("Invalid job format, missing img_name or chat_id")
                    continue

                logger.info(f'Processing job: {prediction_id}, Image: {img_name}, Chat ID: {chat_id}')

                # --- Download Image from S3 ---
                local_img_path = f'static/data/{prediction_id}/{img_name}'
                os.makedirs(os.path.dirname(local_img_path), exist_ok=True)

                s3_client.download_file(images_bucket, img_name, local_img_path)
                logger.info(f'Downloaded {img_name} from S3')

                # --- Run YOLOv5 Object Detection ---
                model = ultralytics.YOLO("yolov5s.pt")
                results = model(local_img_path)

                # Save results to a directory for further processing
                results.save(save_dir=f"static/data/{prediction_id}")

                # --- Path for Predictions ---
                predicted_img_path = Path(f'static/data/{prediction_id}/{img_name}')
                predicted_s3_key = f'predictions/{prediction_id}/{img_name}'

                # --- Upload Predicted Image to S3 ---
                s3_client.upload_file(str(predicted_img_path), images_bucket, predicted_s3_key)
                logger.info(f'Uploaded predicted image to S3: {predicted_s3_key}')

                # --- Parse YOLOv5 Results ---
                pred_summary_path = Path(f'static/data/{prediction_id}/labels/{img_name.split(".")[0]}.txt')

