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
db_name = "config"
collection_name = "image_collection"

# --- MongoDB Connection ---
mongo_client = MongoClient(mongo_uri)
db = mongo_client[db_name]
collection = db[collection_name]

def save_results(prediction_id, labels):
    try:
        collection.insert_one({'predictionId': prediction_id, 'labels': labels})
        logger.info(f"Saved prediction {prediction_id} to MongoDB.")
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {e}")

def process_image(image_path):
    logger.info(f"Processing image: {image_path}")
    results = model.predict(image_path)
    labels = [{'class': det['name'], 'confidence': det['confidence']} for det in results]
    prediction_id = str(uuid.uuid4())
    save_results(prediction_id, labels)
    return prediction_id, labels

if __name__ == "__main__":
    logger.info("YOLOv5 service started.")
