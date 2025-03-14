import time
import ultralytics
import os
import json
import boto3
import requests
import uuid
import yaml
import sys
from pathlib import Path
from loguru import logger
from pymongo import MongoClient, errors
from pymongo.write_concern import WriteConcern
import urllib3
import pymongo

pymongo.common.VALIDATORS["replicaSet"] = lambda x: True  # Avoid strict validation issues

# --- AWS Secrets Manager Setup ---
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")

region_name = "eu-north-1"

# --- Load Secrets from AWS Secrets Manager ---
def load_secrets():
    """Retrieve all secrets from AWS Secrets Manager"""
    try:
        secrets_client = boto3.client("secretsmanager", region_name=region_name)
        response = secrets_client.get_secret_value(SecretId="polybot-secrets")
        secrets = json.loads(response["SecretString"])
        logger.info("Loaded secrets from AWS Secrets Manager")
        return secrets
    except Exception as e:
        logger.error(f"Failed to load secrets from AWS Secrets Manager: {e}")
        raise

# Load secrets once
secrets = load_secrets()

# --- Environment Variables ---
os.environ["S3_BUCKET_NAME"] = secrets.get("S3_BUCKET_NAME", "")
os.environ["SQS_QUEUE_URL"] = secrets.get("SQS_QUEUE_URL", "")
os.environ["TELEGRAM_APP_URL"] = secrets.get("TELEGRAM_APP_URL", "")

images_bucket = os.getenv("S3_BUCKET_NAME")
queue_url = os.getenv("SQS_QUEUE_URL")
polybot_url = os.getenv("POLYBOT_URL")

# --- AWS Clients ---
sqs_client = boto3.client(
    "sqs",
    aws_access_key_id=secrets.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=secrets.get("AWS_SECRET_ACCESS_KEY"),
    region_name=region_name,
)
s3_client = boto3.client(
    "s3",
    aws_access_key_id=secrets.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=secrets.get("AWS_SECRET_ACCESS_KEY"),
    region_name=region_name,
)

# --- MongoDB Connection with Retry ---
def connect_to_mongo():
    mongo_uri = secrets.get("MONGO_URI")
    mongo_client = MongoClient(mongo_uri, retryWrites=True, retryReads=True, serverSelectionTimeoutMS=30000)
    logger.info(f"Initial servers: {mongo_client.nodes}")
    if not mongo_uri:
        logger.error("MONGO_URI is missing from secrets")
        raise ValueError("MONGO_URI is missing from AWS Secrets Manager")
    logger.info(f"Using MONGO_URI before connection: {mongo_uri}")
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt} with MONGO_URI: {mongo_uri}")
            mongo_client = MongoClient(mongo_uri, retryWrites=True, retryReads=True)
            db = mongo_client[secrets.get("MONGO_DB", "config")]
            collection = db[secrets.get("MONGO_COLLECTION", "image_collection")]
            mongo_client.admin.command("ping")
            logger.info("Connected to MongoDB successfully")
            return collection
        except (errors.ConnectionFailure, errors.NotPrimaryError) as e:
            logger.error(f"MongoDB connection attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    logger.error("MongoDB connection failed after retries")
    raise ConnectionError("Could not connect to MongoDB after multiple retries")

# Initialize MongoDB connection
collection = connect_to_mongo()

# --- Load Class Names ---
def load_class_names():
    """Load COCO class names from YAML file."""
    try:
        with open("data/coco128.yaml", "r") as stream:
            names = yaml.safe_load(stream)["names"]
        logger.info("Loaded class names successfully")
        return names
    except Exception as e:
        logger.error(f"Failed to load class names: {e}")
        raise

names = load_class_names()

# --- Load YOLO Model with Download Check ---
model_path = "yolov5su.pt"
model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5su.pt"

def download_model(url, filepath):
    """Download the YOLO model if it doesn't exist."""
    if not os.path.exists(filepath):
        logger.info(f"Downloading model from {url} to {filepath}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    downloaded = min(total_size, f.tell())
                    logger.debug(f"Downloaded {downloaded / 1024 / 1024:.2f} MB / {total_size / 1024 / 1024:.2f} MB")
        logger.info(f"Model downloaded successfully to {filepath}")
    else:
        logger.info(f"Model already exists at {filepath}, skipping download.")

# Download model if not present
download_model(model_url, model_path)

# Load YOLO model
model = ultralytics.YOLO(model_path)

# --- Process SQS Job ---
def process_job(message, receipt_handle):
    """Process an image detection job from SQS."""
    try:
        logger.info(f"Received SQS message: {message['Body']}")
        job = json.loads(message["Body"])
        img_name = job.get("imgName")
        chat_id = job.get("chat_id")
        prediction_id = str(uuid.uuid4())

        if not img_name or not chat_id:
            logger.error("Invalid job format: missing img_name or chat_id")
            sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            return

        logger.info(f"Processing job: {prediction_id}, Image: {img_name}, Chat ID: {chat_id}")

        # Download Image from S3
        local_img_dir = Path(f"static/data/{prediction_id}")
        local_img_dir.mkdir(parents=True, exist_ok=True)
        local_img_path = local_img_dir / img_name

        try:
            s3_client.download_file(images_bucket, img_name, str(local_img_path))
            logger.info(f"Downloaded {img_name} from S3")
        except Exception as e:
            logger.error(f"Failed to download image from S3: {e}")
            return

        # Run YOLOv5 Object Detection
        results = model.predict(
            str(local_img_path),
            save=True,
            save_txt=True,
            project="static/data",
            name=prediction_id,
            exist_ok=True
        )
        labels_dir = local_img_dir / "labels"
        if labels_dir.exists() and labels_dir.is_dir():
            logger.info(f"Labels directory exists: {labels_dir}")
        else:
            logger.error(f"Labels directory does not exist: {labels_dir}")
            labels_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created labels directory: {labels_dir}")

        # Upload Predictions to S3
        predicted_s3_key = f"predictions/{prediction_id}/{img_name}"
        try:
            s3_client.upload_file(str(local_img_path), images_bucket, predicted_s3_key)
            logger.info(f"Uploaded predicted image to S3: {predicted_s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload predicted image to S3: {e}")
            return

        # Parse YOLO Results
        pred_summary_path = Path(f"static/data/{prediction_id}/labels/{Path(img_name).stem}.txt")
        labels = []
        if pred_summary_path.exists():
            with open(pred_summary_path, 'r') as f:
                lines = f.read().splitlines()
                logger.info(f"Prediction file contents: {lines}")
                for line in lines:
                    if line.strip():  # Skip empty lines
                        l = line.split(" ")
                        labels.append({
                            "class": names[int(l[0])],
                            "cx": float(l[1]),
                            "cy": float(l[2]),
                            "width": float(l[3]),
                            "height": float(l[4]),
                        })
        else:
            logger.error(f"Prediction file not found: {pred_summary_path}")

        # Store Prediction in MongoDB
        prediction_summary = {
            "_id": prediction_id,
            "chat_id": chat_id,
            "original_img_path": img_name,
            "predicted_img_path": predicted_s3_key,
            "labels": labels,
            "time": time.time(),
        }
        max_retries = 5
        for attempt in range(max_retries):
            try:
                collection.with_options(write_concern=WriteConcern("majority")).insert_one(prediction_summary)
                logger.info(f"Prediction summary stored: {prediction_summary}")
                logger.info(f"Stored prediction in MongoDB: {prediction_id}")
                break
            except (errors.NotPrimaryError, errors.ServerSelectionTimeoutError) as e:
                logger.error(f"Retry {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        else:
            logger.error("Failed to store prediction in MongoDB after all retries")

        # Notify Polybot
        try:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.info(f"Notifying Polybot at: {polybot_url}")
            logger.info(f"Polybot URL: {polybot_url}")
            polybot_response = requests.post(polybot_url, json={"predictionId": prediction_id}, timeout=10, verify=False)
            if polybot_response.status_code == 200:
                logger.info(f"Polybot notified successfully: {polybot_url}")
            else:
                logger.error(f"Polybot notification failed with status: {polybot_response.status_code}")
        except Exception as e:
            logger.error(f"Error notifying Polybot: {e}")

        # Delete Message from SQS
        sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
        logger.info(f"Job {prediction_id} completed and removed from SQS")

    except Exception as e:
        logger.error(f"Error processing job: {e}")
        time.sleep(1)

# --- Main Consumer Loop ---
def consume():
    """Polls SQS queue for messages and processes image jobs."""
    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=5
            )
            if "Messages" not in response:
                logger.info("No messages in SQS queue. Waiting...")
                time.sleep(10)
                continue

            message = response["Messages"][0]
            receipt_handle = message["ReceiptHandle"]
            process_job(message, receipt_handle)
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(1)

# Start consumer
if __name__ == "__main__":
    consume()