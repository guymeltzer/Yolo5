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
from pymongo import MongoClient, errors

# --- AWS Secrets Manager Setup ---
try:
    secrets_client = boto3.client('secretsmanager', region_name="eu-north-1")
    response = secrets_client.get_secret_value(SecretId="polybot-secrets")
    secrets = json.loads(response['SecretString'])
    logger.info("Loaded secrets from AWS Secrets Manager")
except Exception as e:
    logger.error(f"Failed to load secrets from AWS Secrets Manager: {e}")
    exit(1)

# --- Set Environment Variables from Secrets ---
os.environ["S3_BUCKET_NAME"] = secrets.get("S3_BUCKET_NAME", "")
os.environ["SQS_QUEUE_URL"] = secrets.get("SQS_QUEUE_URL", "")
os.environ["MONGO_URI"] = secrets.get("MONGO_URI", "")

# --- Environment Variables ---
images_bucket = os.getenv('S3_BUCKET_NAME')
queue_url = os.getenv('SQS_QUEUE_URL')
mongo_uri = os.getenv('MONGO_URI')

if not images_bucket or not queue_url or not mongo_uri:
    logger.error("Missing required environment variables. Exiting...")
    exit(1)

db_name = os.getenv('MONGO_DB', 'default_db')
collection_name = os.getenv('MONGO_COLLECTION', 'predictions')
polybot_url = os.getenv('POLYBOT_URL', 'http://polybot-service:30619/results')

# --- AWS Clients ---
sqs_client = boto3.client('sqs', region_name='eu-north-1')
s3_client = boto3.client('s3')

# --- MongoDB Connection with Retry ---
max_retries = 5
for attempt in range(1, max_retries + 1):
    try:
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = mongo_client[db_name]
        collection = db[collection_name]
        mongo_client.admin.command('ping')  # Verify connection
        logger.info("Connected to MongoDB successfully")
        break
    except errors.ConnectionFailure as e:
        logger.error(f"MongoDB connection attempt {attempt} failed: {e}")
        time.sleep(5)  # Retry after delay
else:
    logger.error("MongoDB connection failed after retries. Exiting...")
    exit(1)

# --- Load Class Names ---
try:
    with open("data/coco128.yaml", "r") as stream:
        names = yaml.safe_load(stream)['names']
    logger.info("Loaded class names successfully")
except Exception as e:
    logger.error(f"Failed to load class names: {e}")
    exit(1)

def consume():
    """Polls SQS queue for messages and processes image jobs."""
    try:
        model = ultralytics.YOLO("yolov5s.pt")  # Load YOLO model once
        logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        exit(1)

    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=5
            )

            if 'Messages' not in response:
                logger.info("No messages in SQS queue. Waiting...")
                time.sleep(10)  # Wait before polling again
                continue

            message = response['Messages'][0]
            receipt_handle = message['ReceiptHandle']

            # Parse job details
            try:
                job = json.loads(message['Body'])
                img_name = job.get('img_name')
                chat_id = job.get('chat_id')
                prediction_id = str(uuid.uuid4())

                if not img_name or not chat_id:
                    logger.error("Invalid job format: missing img_name or chat_id")
                    sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
                    continue
            except json.JSONDecodeError:
                logger.error("Invalid JSON format in SQS message")
                sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
                continue

            logger.info(f'Processing job: {prediction_id}, Image: {img_name}, Chat ID: {chat_id}')

            # --- Download Image from S3 ---
            local_img_dir = Path(f'static/data/{prediction_id}')
            local_img_dir.mkdir(parents=True, exist_ok=True)
            local_img_path = local_img_dir / img_name

            try:
                s3_client.download_file(images_bucket, img_name, str(local_img_path))
                logger.info(f'Downloaded {img_name} from S3')
            except Exception as e:
                logger.error(f"Failed to download image from S3: {e}")
                continue

            # --- Run YOLOv5 Object Detection ---
            results = model.predict(str(local_img_path), save=True, save_dir=str(local_img_dir))

            # --- Upload Predictions to S3 ---
            predicted_s3_key = f'predictions/{prediction_id}/{img_name}'

            try:
                s3_client.upload_file(str(local_img_path), images_bucket, predicted_s3_key)
                logger.info(f'Uploaded predicted image to S3: {predicted_s3_key}')
            except Exception as e:
                logger.error(f"Failed to upload prediction to S3: {e}")

            # --- Parse YOLO Results ---
            pred_summary_path = local_img_dir / f'labels/{Path(img_name).stem}.txt'

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

            try:
                collection.insert_one(prediction_summary)
                logger.info(f'Stored prediction in MongoDB: {prediction_id}')
            except Exception as e:
                logger.error(f"Failed to store prediction in MongoDB: {e}")

            # --- Notify Polybot ---
            try:
                polybot_response = requests.post(polybot_url, json={"predictionId": prediction_id})
                if polybot_response.status_code == 200:
                    logger.info(f'Polybot notified successfully for {prediction_id}')
                else:
                    logger.error(f'Failed to notify Polybot: {polybot_response.text}')
            except Exception as e:
                logger.error(f"Error notifying Polybot: {e}")

            # --- Delete Message from SQS ---
            sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            logger.info(f'Job {prediction_id} completed and removed from SQS')

        except Exception as e:
            logger.error(f'Error processing job: {e}')
            time.sleep(1)

# Start consumer
consume()
