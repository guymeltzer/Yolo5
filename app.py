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

# --- AWS Secrets Manager Setup ---
secrets_client = boto3.client('secretsmanager', region_name="eu-north-1")
response = secrets_client.get_secret_value(SecretId="polybot-secrets")
secrets = json.loads(response['SecretString'])

# --- Set Environment Variables from Secrets ---
os.environ["S3_BUCKET_NAME"] = secrets["S3_BUCKET_NAME"]
os.environ["SQS_QUEUE_URL"] = secrets["SQS_QUEUE_URL"]
os.environ["MONGO_URI"] = secrets["MONGO_URI"]

# --- Environment Variables ---
images_bucket = os.environ['S3_BUCKET_NAME']
queue_url = os.environ['SQS_QUEUE_URL']
mongo_uri = os.environ['MONGO_URI']
db_name = os.environ.get('MONGO_DB', 'default_db')
collection_name = os.environ.get('MONGO_COLLECTION', 'predictions')
polybot_url = os.environ.get('POLYBOT_URL', 'http://polybot-service:30619/results')

# --- AWS Clients ---
sqs_client = boto3.client('sqs', region_name='eu-north-1')
s3_client = boto3.client('s3')

# --- MongoDB Client with Retry Logic ---
for _ in range(5):  # Retry up to 5 times
    try:
        mongo_client = MongoClient(mongo_uri)
        db = mongo_client[db_name]
        collection = db[collection_name]
        logger.info("Connected to MongoDB successfully")
        break  # Exit loop if successful
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        time.sleep(5)  # Wait before retrying
else:
    logger.error("MongoDB connection failed after retries, exiting...")
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
    model = ultralytics.YOLO("yolov5s.pt")  # Load YOLO model once

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
                results = model.predict(local_img_path, save=True, save_dir=f"static/data/{prediction_id}")

                # --- Path for Predictions ---
                predicted_img_path = Path(f'static/data/{prediction_id}/{img_name}')
                predicted_s3_key = f'predictions/{prediction_id}/{img_name}'

                # --- Upload Predicted Image to S3 ---
                s3_client.upload_file(str(predicted_img_path), images_bucket, predicted_s3_key)
                logger.info(f'Uploaded predicted image to S3: {predicted_s3_key}')

                # --- Parse YOLOv5 Results ---
                pred_summary_path = Path(f'static/data/{prediction_id}/labels/{Path(img_name).stem}.txt')

                labels = []
                if pred_summary_path.exists():
                    with open(pred_summary_path) as f:
                        for line in f.read().splitlines():
                            l = line.split(' ')
                            labels.append({
                                'class': names[int(l[0])],
                                'cx': float(l[1]),
                                'cy': float(l[2]),
                                'width': float(l[3]),
                                'height': float(l[4]),
                            })

                # --- Store Prediction in MongoDB ---
                prediction_summary = {
                    '_id': prediction_id,
                    'chat_id': chat_id,
                    'original_img_path': img_name,
                    'predicted_img_path': predicted_s3_key,
                    'labels': labels,
                    'time': time.time()
                }
                collection.insert_one(prediction_summary)
                logger.info(f'Stored prediction in MongoDB: {prediction_id}')

                # --- Notify Polybot ---
                polybot_response = requests.post(polybot_url, json={"predictionId": prediction_id})
                if polybot_response.status_code == 200:
                    logger.info(f'Polybot notified successfully for {prediction_id}')
                else:
                    logger.error(f'Failed to notify Polybot for {prediction_id}: {polybot_response.text}')

                # --- Delete Message from SQS ---
                sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
                logger.info(f'Job {prediction_id} completed and removed from SQS')

            else:
                # If no messages, wait and try again
                logger.info("No messages in SQS queue. Waiting...")
                time.sleep(10)  # Wait for 10 seconds before checking again

        except Exception as e:
            logger.error(f'Error processing job: {e}')
            time.sleep(1)  # Brief delay before retrying


