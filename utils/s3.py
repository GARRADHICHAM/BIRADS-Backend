# app/utils/s3.py
import boto3
from botocore.exceptions import ClientError
import os
from typing import BinaryIO
import io

# AWS S3 connection
from dotenv import load_dotenv
load_dotenv()

# AWS S3 connection
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name='eu-north-1'
)
BUCKET_NAME = "semi-auto-classification"

def upload_to_s3(file_obj: BinaryIO, filename: str, folder: str = "") -> str:
    """
    Upload a file to S3 bucket
    
    Args:
        file_obj: File-like object to upload
        filename: Name of the file
        folder: Folder path in S3 bucket
        
    Returns:
        str: S3 path of uploaded file
    """
    try:
        # Create the full S3 key
        s3_key = f"{folder}/{filename}" if folder else filename
        
        # Reset file pointer to beginning
        file_obj.seek(0)
        
        # Upload file to S3
        s3_client.upload_fileobj(
            file_obj,
            BUCKET_NAME,
            s3_key,
            ExtraArgs={
                'ContentType': 'image/png',  # Adjust based on your image type
                'ServerSideEncryption': 'AES256'  # Optional: encrypt at rest
            }
        )
        
        # Return the S3 path
        s3_path = f"s3://{BUCKET_NAME}/{s3_key}"
        return s3_path
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            raise Exception(f"S3 bucket '{BUCKET_NAME}' does not exist")
        else:
            raise Exception(f"Failed to upload to S3: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error uploading to S3: {str(e)}")

def get_s3_url(s3_path: str) -> str:
    """
    Convert S3 path to public URL
    
    Args:
        s3_path: S3 path (s3://bucket/key)
        
    Returns:
        str: Public S3 URL
    """
    # Extract bucket and key from s3 path
    if s3_path.startswith('s3://'):
        path_parts = s3_path[5:].split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        
        # Return public URL
        region = os.getenv('AWS_REGION', 'us-east-1')
        return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    
    return s3_path

def delete_from_s3(s3_path: str) -> bool:
    """
    Delete file from S3
    
    Args:
        s3_path: S3 path to delete
        
    Returns:
        bool: Success status
    """
    try:
        # Extract key from s3 path
        if s3_path.startswith('s3://'):
            path_parts = s3_path[5:].split('/', 1)
            key = path_parts[1] if len(path_parts) > 1 else ''
            
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
            return True
    except Exception as e:
        print(f"Error deleting from S3: {str(e)}")
        return False

def get_presigned_url(key, expiration=3600):
    """Generate a presigned URL for an S3 object"""
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        return None