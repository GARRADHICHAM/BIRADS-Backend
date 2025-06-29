# app/database.py
import pymongo
import os
from bson.objectid import ObjectId

# Database connection
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["mammogram_db"]
users_collection = db["users"]
images_collection = db["images"]
annotations_collection = db["annotations"]


# Function to get the database for dependency injection
def get_database():
    return db

# Function to get specific collections
def get_users_collection():
    return users_collection

def get_images_collection():
    return images_collection

def get_annotations_collection():
    return annotations_collection



IMAGES_SCHEMA = """
{
  "_id": ObjectId,
  "original_path": String,  # S3 path
  "mask_path": String,      # AI-generated mask S3 path
  "confirmed_mask_path": String,  # Doctor-confirmed mask S3 path
  "birads_score": Number,
  "lesion_type": String,
  "uploaded_by": String,
  "uploaded_at": Date,
  "status": String,  # "pending", "annotated", "confirmed"
  "confirmed_by": String,
  "confirmed_at": Date
}
"""

ANNOTATIONS_SCHEMA = """
{
  "_id": ObjectId,
  "image_id": String,
  "mask_path": String,
  "birads_score": Number,
  "lesion_type": String,
  "annotated_by": String,
  "annotated_at": Date
}
"""

