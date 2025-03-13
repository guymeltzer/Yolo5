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
import os
import boto3
import json
import time
from pymongo import MongoClient, errors
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
        raise  # Raising the exception instead of exiting


# Load secrets once
secrets = load_secrets()

# --- Environment Variables ---
os.environ["S3_BUCKET_NAME"] = secrets.get("S3_BUCKET_NAME", "")
os.environ["SQS_QUEUE_URL"] = secrets.get("SQS_QUEUE_URL", "")

images_bucket = os.getenv("S3_BUCKET_NAME")
queue_url = os.getenv("SQS_QUEUE_URL")
polybot_url = os.getenv("TELEGRAM_APP_URL")

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
    """Connect to MongoDB using URI from AWS Secrets Manager."""
    mongo_uri = "mongodb://mongodb.mongodb.svc.cluster.local:27017/?replicaset=rs0&readPreference=primary"
    # Fetch directly from loaded secrets

    if not mongo_uri:
        logger.error("MONGO_URI is missing from secrets. Exiting...")
        raise ValueError("MONGO_URI is missing from AWS Secrets Manager")



    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Using MONGO_URI: {mongo_uri}")
            mongo_client = MongoClient(mongo_uri, directConnection=True)
            db = mongo_client["config"]
            collection = db["image_collection"]
            mongo_client.admin.command("ping")  # Verify connection
            logger.info("Connected to MongoDB successfully")
            return collection
        except errors.ConnectionFailure as e:
            logger.error(f"MongoDB connection attempt {attempt} failed: {e}")
            time.sleep(2**attempt)  # Exponential backoff

    logger.error("MongoDB connection failed after retries. Exiting...")
    raise ConnectionError("Could not connect to MongoDB after multiple retries")


# Initialize MongoDB connection
collection = connect_to_mongo()


# --- Load Class Names ---
def load_class_names():
    try:
        with open("data/coco128.yaml", "r") as stream:
            names = yaml.safe_load(stream)["names"]
        logger.info("Loaded class names successfully")
        return names
    except Exception as e:
        logger.error(f"Failed to load class names: {e}")
        exit(1)


names = load_class_names()


def process_job(message, receipt_handle):
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

        logger.info(
            f"Processing job: {prediction_id}, Image: {img_name}, Chat ID: {chat_id}"
        )

        # --- Download Image from S3 ---
        local_img_dir = Path(f"static/data/{prediction_id}")
        local_img_dir.mkdir(parents=True, exist_ok=True)
        local_img_path = local_img_dir / img_name

        try:
            s3_client.download_file(images_bucket, img_name, str(local_img_path))
            logger.info(f"Downloaded {img_name} from S3")
        except Exception as e:
            logger.error(f"Failed to download image from S3: {e}")
            return

        # --- Run YOLOv5 Object Detection ---
        model = ultralytics.YOLO("yolov5s.pt")  # Load YOLO model once
        results = model.predict(
            str(local_img_path), save=True, save_dir=str(local_img_dir)
        )

        # --- Upload Predictions to S3 ---
        predicted_s3_key = f"predictions/{prediction_id}/{img_name}"

        try:
            # Upload predicted image to S3 (not re-downloading)
            s3_client.upload_file(str(local_img_path), images_bucket, predicted_s3_key)
            logger.info(f"Uploaded predicted image to S3: {predicted_s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload predicted image to S3: {e}")
            return

        # --- Parse YOLO Results ---
        pred_summary_path = local_img_dir / f"labels/{Path(img_name).stem}.txt"
        labels = []

        if pred_summary_path.exists():
            with open(pred_summary_path) as f:
                for line in f.read().splitlines():
                    l = line.split(" ")
                    labels.append(
                        {
                            "class": names[int(l[0])],
                            "cx": float(l[1]),
                            "cy": float(l[2]),
                            "width": float(l[3]),
                            "height": float(l[4]),
                        }
                    )

        # --- Store Prediction in MongoDB ---
        prediction_summary = {
            "_id": prediction_id,
            "chat_id": chat_id,
            "original_img_path": img_name,
            "predicted_img_path": predicted_s3_key,
            "labels": labels,
            "time": time.time(),
        }
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                collection.insert_one(prediction_summary)
                logger.info(f"Stored prediction in MongoDB: {prediction_id}")
                break
            except errors.NotWritablePrimary as e:
                logger.error(f"Attempt {attempt}: Failed to store prediction in MongoDB: {e}")
                if attempt < max_retries:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error("Exceeded maximum retries. Could not store prediction in MongoDB.")
            except Exception as e:
                    logger.error(f"Failed to store prediction in MongoDB: {e}")
                    break
        # --- Notify Polybot ---
        try:
            polybot_response = requests.post(
                polybot_url, json={"predictionId": prediction_id}
            )
            if polybot_response.status_code == 200:
              logger.info(f"Polybot URL is reachable: {polybot_url}")
            else:
              logger.error(f"Polybot URL returned unexpected status: {polybot_response.status_code}")
        except Exception as e:
            logger.error(f"Error reaching Polybot URL: {e}")

        # --- Delete Message from SQS ---
        sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
        logger.info(f"Job {prediction_id} completed and removed from SQS")

    except Exception as e:
        logger.error(f"Error processing job: {e}")
        time.sleep(1)


def consume():
    """Polls SQS queue for messages and processes image jobs."""
    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=5
            )

            if "Messages" not in response:
                logger.info("No messages in SQS queue. Waiting...")
                time.sleep(10)  # Wait before polling again
                continue

            message = response["Messages"][0]
            receipt_handle = message["ReceiptHandle"]
            process_job(message, receipt_handle)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(1)


# Start consumer
consume()
