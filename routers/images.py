# app/routers/images.py
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException ,FastAPI,  Query, APIRouter
from schemas.models import User, ImageMetadata
from fastapi.responses import Response
from database import images_collection , annotations_collection
from utils.s3 import upload_to_s3, get_presigned_url
# from models.ai_model import model
from datetime import datetime
from bson.objectid import ObjectId
from pydantic import BaseModel
from scipy import ndimage
from fastapi import Query
from fastapi.responses import StreamingResponse
import pydicom
from PIL import Image
import numpy as np
import base64
import requests
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os


from skimage import morphology

# from models.ai_model import model

import boto3
import os

def download_model_from_s3(bucket_name, object_key, local_path):
    s3 = boto3.client('s3')
    if not os.path.exists(local_path):
        s3.download_file(bucket_name, object_key, local_path)

router = APIRouter()





import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

# First, recreate your U-Net architecture
def build_unet_model(input_shape=(256, 256, 1)):
    """Build a U-Net model"""
    inputs = Input(input_shape)

    # Encoder (Downsampling)
    # Block 1
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Block 2
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 3
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Block 4
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bridge
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder (Upsampling)
    # Block 6
    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    # Block 7
    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    # Block 8
    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    # Block 9
    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create the model with the same architecture
model2 = build_unet_model(input_shape=(256, 256, 1))  # Make sure this matches your original input shape

# Compile the model (use the same loss and optimizer you used during training)
model2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# # Load only the weights
# try:
#     model2.load_weights("E:/Segmentation-BIRADS/BackEnd/routers/unet_mammography_best_model.h5")
#     print("Model weights loaded successfully!")
# except:
#     # If loading the whole weights file doesn't work, try loading just the weights
#     # This is for TensorFlow 2.x compatibility
#     try:
#         model.load_weights("unet_mammography_best_model.h5", by_name=True, skip_mismatch=True)
#         print("Model weights loaded with name mapping!")
#     except Exception as e:
#         print(f"Error loading weights: {e}")
        
# Set your S3 bucket and key
S3_BUCKET = "aimodel-seg"
S3_KEY = "https://aimodel-seg.s3.eu-north-1.amazonaws.com/unet_mammography_best_model.h5"
LOCAL_MODEL_PATH = "routers/unet_mammography_best_model.h5"

# Download the model from S3 if not present
download_model_from_s3(S3_BUCKET, S3_KEY, LOCAL_MODEL_PATH)

# Build and load the model as before
model2 = build_unet_model(input_shape=(256, 256, 1))
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.load_weights(LOCAL_MODEL_PATH)

@router.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    
):
    # Read image content
    image_content = await file.read()
    image = Image.open(io.BytesIO(image_content))
    
    # Generate unique filename
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    
    # Upload to S3
    original_path = f"original/{filename}"
    upload_result = upload_to_s3(
        image_content,
        original_path,
        file.content_type
    )
    
    if not upload_result:
        raise HTTPException(status_code=500, detail="Failed to upload to S3")
    
    # Run AI model for segmentation
    image_array = np.array(image.resize((256, 256)))
    image_array = np.expand_dims(image_array, axis=0) / 255.0
    
    mask_pred = model2.predict(image_array)[0]
    mask_image = Image.fromarray((mask_pred * 255).astype(np.uint8))
    
    # Save mask to S3
    mask_buffer = io.BytesIO()
    mask_image.save(mask_buffer, format="PNG")
    mask_buffer.seek(0)
    
    mask_path = f"masks/{filename.split('.')[0]}_mask.png"
    mask_upload_result = upload_to_s3(
        mask_buffer.getvalue(),
        mask_path,
        "image/png"
    )
    
    if not mask_upload_result:
        raise HTTPException(status_code=500, detail="Failed to upload mask to S3")
    
    # Save metadata to MongoDB
    image_metadata = ImageMetadata(
        original_path=original_path,
        mask_path=mask_path,
        uploaded_by='me',
        uploaded_at=datetime.now(),
        status="pending"
    )
    
    image_id = images_collection.insert_one(image_metadata.dict()).inserted_id
    
    return {"image_id": str(image_id), "status": "success"}

@router.get("/images/{image_id}")
async def get_image(
    image_id: str,
    
):
    # Get image metadata from MongoDB
    image_data = images_collection.find_one({"_id": ObjectId(image_id)})
    
    if not image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Generate presigned URLs for the original image and mask
    original_url = get_presigned_url(image_data['original_path'])
    mask_url = get_presigned_url(image_data['mask_path'])
    
    if not original_url or not mask_url:
        raise HTTPException(status_code=500, detail="Failed to generate presigned URLs")
    
    return {
        "imageId": str(image_data["_id"]),
        "originalUrl": original_url,
        "maskUrl": mask_url,
        "biradsScore": image_data.get("birads_score"),
        "lesionType": image_data.get("lesion_type"),
        "status": image_data["status"]
    }

@router.get("/next-image")
async def get_next_image(
   
):
    # Find the next pending image
    next_image = images_collection.find_one({"status": "pending"})
    
    if not next_image:
        return {"status": "complete", "message": "No more images to annotate"}
    
    return {"imageId": str(next_image["_id"])}


@router.get("/images-all/")
async def get_all_images(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of items to return"),
):
    # Create a filter for PNG files using regex
    png_filter = {"imagePath": {"$regex": "\.png$"}}
    
    # Get total count for pagination info (only PNG files)
    total_count = images_collection.count_documents(png_filter)
    
    # Get paginated image metadata from MongoDB (only PNG files)
    all_images_data = images_collection.find(png_filter).skip(skip).limit(limit)
    
    if not all_images_data:
        return {"total": total_count, "images": []}
    
    results = []
    for image_data in all_images_data:
        # Generate presigned URL for each image
        original_url = image_data['imagePath']
        
        # Add image info to results
        results.append({
            "imageId": str(image_data["_id"]),
            "originalUrl": original_url,
        })
    
    # Return paginated response with total count
    return {
        "total": total_count,
        "images": results,
        "skip": skip,
        "limit": limit
    }

def post_process_predictions(mask, threshold=0.5, min_size=50):
    """
    Apply post-processing to refine prediction masks.
    """
    # Threshold
    binary = (mask > threshold).astype(np.uint8)
    
    # Remove small objects
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=min_size)
    
    # Fill holes
    binary = ndimage.binary_fill_holes(binary)
    
    # Apply closing to smooth boundaries
    binary = morphology.binary_closing(binary, morphology.disk(2))
    
    return binary * 255    
    

    
    


import base64
import io
import numpy as np
import requests
from PIL import Image
from fastapi import HTTPException, Query
from bson import ObjectId

@router.get("/mammogram/predict")
async def predict_mammogram_mask(
    current_index: int = Query(0, ge=0, description="Current image index"),
    case_id: str = Query(None, description="Optional case ID to filter")
):
    """
    Endpoint to retrieve the next mammogram image URL and generate its predicted mask.
    """
    try:
        # Prepare query filter
        query_filter = {"imagePath": {"$regex": "\.png$"}}
        if case_id:
            query_filter["caseId"] = case_id
        
        # Count total images
        total_images = images_collection.count_documents(query_filter)
        
        # Ensure index is within bounds
        next_index = current_index + 1
        if next_index >= total_images:
            next_index = 0  # Wrap around to first image
        
        # Retrieve the image document
        image_data = images_collection.find(query_filter).skip(next_index).limit(1).next()
        
        # Get the original image URL
        original_url = image_data['imagePath']
        
        # Download image
        response = requests.get(original_url)
        original_image = Image.open(io.BytesIO(response.content))
        
        # Convert to grayscale if not already
        if original_image.mode != 'L':
            original_image = original_image.convert('L')
        
        # Resize to 512x512 to match model input
        resized_image = original_image.resize((512, 512))
        
        # Convert to numpy array and normalize
        img_array = np.array(resized_image)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=[0, -1])
        
        # Generate prediction
        prediction = model2.predict(img_array)
        
        # Post-process the prediction
        mask = post_process_predictions(prediction[0, :, :, 0])
        
        # Convert mask to PNG image
        mask_image = Image.fromarray(mask.astype(np.uint8))
        
        # Convert mask to base64
        mask_byte_arr = io.BytesIO()
        mask_image.save(mask_byte_arr, format='PNG')
        mask_base64 = base64.b64encode(mask_byte_arr.getvalue()).decode('utf-8')
        
        # Apply Viridis Colormap
        # Use matplotlib for Viridis conversion
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Normalize the grayscale image
        img_normalized = (img_array[0, :, :, 0] - img_array[0, :, :, 0].min()) / (img_array[0, :, :, 0].max() - img_array[0, :, :, 0].min())
        
        # Apply Viridis colormap
        viridis_img = cm.viridis(img_normalized)
        
        # Convert to 8-bit image
        viridis_img_8bit = (viridis_img[:, :, :3] * 255).astype(np.uint8)
        
        # Convert to PIL Image
        viridis_pil = Image.fromarray(viridis_img_8bit)
        
        # Convert Viridis image to base64
        viridis_byte_arr = io.BytesIO()
        viridis_pil.save(viridis_byte_arr, format='PNG')
        viridis_base64 = base64.b64encode(viridis_byte_arr.getvalue()).decode('utf-8')
        
        # Return response
        return {
            "originalImageUrl": original_url,
            "predictedMaskBase64": mask_base64,
            "viridisImageBase64": viridis_base64,  # New field for Viridis colormap
            "totalImages": total_images,
            "currentIndex": next_index,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.get("/mammogram/predict_next")
async def predict_mammogram_mask(
    current_index: int = Query(0, ge=0, description="Current image index"),
    case_id: str = Query(None, description="Optional case ID to filter")
):
    """
    Endpoint to retrieve the next mammogram image URL and generate its predicted mask.
    """
    try:
        # Prepare query filter
        query_filter = {"imagePath": {"$regex": "\.png$"}}
        if case_id:
            query_filter["caseId"] = case_id
        
        # Count total images
        total_images = images_collection.count_documents(query_filter)
        
        # Ensure index is within bounds
        next_index = current_index + 1
        if next_index >= total_images:
            next_index = 0  # Wrap around to first image
        
        # Retrieve the image document
        image_data = images_collection.find(query_filter).skip(next_index).limit(1).next()
        
        # Get the original image URL
        original_url = image_data['imagePath']
        id = str(image_data['_id'])
        classe = image_data['folderName']
        filename = image_data['fileName']
        uploadDate = image_data['uploadDate']
        
        # Download image
        response = requests.get(original_url)
        original_image = Image.open(io.BytesIO(response.content))
        
        # Convert to grayscale if not already
        if original_image.mode != 'L':
            original_image = original_image.convert('L')
        
        # Resize to 512x512 to match model input
        resized_image = original_image.resize((256, 256))
        
        # Convert to numpy array and normalize
        img_array = np.array(resized_image)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=[0, -1])
        
        # Generate prediction
        prediction = model2.predict(img_array)
        
        # Post-process the prediction
        mask = post_process_predictions(prediction[0, :, :, 0])
        
        # Convert mask to PNG image
        mask_image = Image.fromarray(mask.astype(np.uint8))
        
        # Convert mask to base64
        mask_byte_arr = io.BytesIO()
        mask_image.save(mask_byte_arr, format='PNG')
        mask_base64 = base64.b64encode(mask_byte_arr.getvalue()).decode('utf-8')
        
        # # Apply Viridis Colormap
        # # Use matplotlib for Viridis conversion
        # import matplotlib.pyplot as plt
        # import matplotlib.cm as cm
        
        # # Normalize the grayscale image
        # img_normalized = (img_array[0, :, :, 0] - img_array[0, :, :, 0].min()) / (img_array[0, :, :, 0].max() - img_array[0, :, :, 0].min())
        
        # # Apply Viridis colormap
        # viridis_img = cm.viridis(img_normalized)
        
        # # Convert to 8-bit image
        # viridis_img_8bit = (viridis_img[:, :, :3] * 255).astype(np.uint8)
        
        # # Convert to PIL Image
        # viridis_pil = Image.fromarray(viridis_img_8bit)
        
        # # Convert Viridis image to base64
        # viridis_byte_arr = io.BytesIO()
        # viridis_pil.save(viridis_byte_arr, format='PNG')
        # viridis_base64 = base64.b64encode(viridis_byte_arr.getvalue()).decode('utf-8')
        
        # Return response
        return {
            "originalImageUrl": original_url,
            "predictedMaskBase64": mask_base64,
            "totalImages": total_images,
            "currentIndex": next_index,
            "id":id,
            "classe":classe,
            "fileName":filename,
            "uploadData":uploadDate,
            
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    
# Add this new endpoint to images.py
@router.get("/api/mammogram/next-unannotated")
async def get_next_unannotated_image(
    current_index: int = Query(0, ge=0, description="Current image index"),
    user_id: str = Query(..., description="User ID to check annotations"),
    case_id: str = Query(None, description="Optional case ID to filter")
):
    """
    Endpoint to retrieve the next unannotated mammogram image for a specific user.
    """
    try:
        # Prepare query filter for images
        query_filter = {"imagePath": {"$regex": "\.png$"}}
        if case_id:
            query_filter["caseId"] = case_id
        
        # Get total number of images
        total_images = images_collection.count_documents(query_filter)
        
        # Start from current_index and look for unannotated image
        for i in range(total_images):
            index = (current_index + i) % total_images  # Wrap around to beginning if needed
            
            # Get image at current index
            image = images_collection.find(query_filter).skip(index).limit(1).next()
            
            # Check if user has already annotated this image
            annotation_exists = annotations_collection.find_one({
                "image_id": image["_id"],
                "annotated_by": user_id
            })
            
            # If no annotation exists for this user, process this image
            if not annotation_exists:
                # Get the image data
                original_url = image["imagePath"]
                id = str(image["_id"])
                classe = image.get("folderName")
                filename = image.get("fileName")
                uploadDate = image.get("uploadDate")

                # Download and process the image for prediction
                response = requests.get(original_url)
                original_image = Image.open(io.BytesIO(response.content))
                
                # Convert to grayscale if not already
                if original_image.mode != 'L':
                    original_image = original_image.convert('L')
                
                # Resize to match model input
                resized_image = original_image.resize((256, 256))
                
                # Convert to numpy array and normalize
                img_array = np.array(resized_image)
                img_array = img_array / 255.0
                
                # Add batch and channel dimensions
                img_array = np.expand_dims(img_array, axis=[0, -1])
                
                # Generate prediction
                prediction = model2.predict(img_array)
                
                # Post-process the prediction
                mask = post_process_predictions(prediction[0, :, :, 0])
                
                # Convert mask to PNG image and then to base64
                mask_image = Image.fromarray(mask.astype(np.uint8))
                mask_byte_arr = io.BytesIO()
                mask_image.save(mask_byte_arr, format='PNG')
                mask_base64 = base64.b64encode(mask_byte_arr.getvalue()).decode('utf-8')

                return {
                    "originalImageUrl": original_url,
                    "predictedMaskBase64": mask_base64,
                    "totalImages": total_images,
                    "currentIndex": index,
                    "id": id,
                    "classe": classe,
                    "fileName": filename,
                    "uploadData": uploadDate,
                    "remainingImages": total_images - index
                }
        
        # If all images have been annotated
        return {
            "message": "All images have been annotated",
            "completed": True,
            "totalImages": total_images
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching next unannotated image: {str(e)}")
# @router.post("/segment-mammogram")
# async def segment_mammogram(file: UploadFile = File(...)):
#     """
#     Endpoint that takes a mammogram image as input and returns the post-processed segmentation mask.
    
#     Args:
#         file: The uploaded mammogram image file
    
#     Returns:
#         The segmentation mask as a PNG image
#     """
#     try:
#         # Validate file type
#         if not file.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="File must be an image")
        
#         # Read and preprocess the image
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents))
        
#         # Convert to grayscale if not already
#         if image.mode != 'L':
#             image = image.convert('L')
        
#         # Resize to 512x512 to match model input
#         image = image.resize((256, 256))
        
#         # Convert to numpy array and normalize
#         img_array = np.array(image)
#         img_array = img_array / 255.0  # Normalize to [0,1]
        
#         # Add batch and channel dimensions
#         img_array = np.expand_dims(img_array, axis=[0, -1])
        
#         # Generate prediction
#         prediction = model2.predict(img_array)
        
#         # Post-process the prediction using your custom function
#         mask = post_process_predictions(prediction[0, :, :, 0])
        
#         # Convert mask to PNG image
#         mask_image = Image.fromarray(mask.astype(np.uint8))
#         img_byte_arr = io.BytesIO()
#         mask_image.save(img_byte_arr, format='PNG')
#         img_byte_arr.seek(0)
        
#         # Return the mask as a PNG image
#         return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/segment-mammogram")
async def segment_mammogram(file: UploadFile = File(...)):
    """
    Endpoint that takes a mammogram image as input and returns the post-processed segmentation mask.
    Supports DICOM files (.dcm) and standard image formats (PNG, JPG, JPEG, etc.).
    
    Args:
        file: The uploaded mammogram image file (DICOM or standard image format)
    
    Returns:
        The segmentation mask as a PNG image
    """
    try:
        # Read file contents
        contents = await file.read()
        
        # Determine file type and process accordingly
        if file.filename.lower().endswith('.dcm') or file.content_type == 'application/dicom':
            # Handle DICOM files
            image = process_dicom_file(contents)
        elif file.content_type.startswith('image/') or file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            # Handle standard image formats
            image = process_standard_image(contents)
        else:
            raise HTTPException(
                status_code=400, 
                detail="File must be a DICOM file (.dcm) or a standard image format (PNG, JPG, JPEG, BMP, TIFF)"
            )
        
        # Common preprocessing for all image types
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to match model input (you had 256x256, but comment mentions 512x512)
        image = image.resize((256, 256))
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=[0, -1])
        
        # Generate prediction
        prediction = model2.predict(img_array)
        
        # Post-process the prediction using your custom function
        mask = post_process_predictions(prediction[0, :, :, 0])
        
        # Convert mask to PNG image
        mask_image = Image.fromarray(mask.astype(np.uint8))
        img_byte_arr = io.BytesIO()
        mask_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Return the mask as a PNG image
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


def process_dicom_file(contents: bytes) -> Image.Image:
    """
    Process DICOM file and convert to PIL Image.
    
    Args:
        contents: Raw bytes of the DICOM file
    
    Returns:
        PIL Image object
    """
    try:
        # Read DICOM file from bytes
        dicom_data = pydicom.dcmread(io.BytesIO(contents))
        
        # Extract pixel array
        pixel_array = dicom_data.pixel_array
        
        # Handle different photometric interpretations
        if hasattr(dicom_data, 'PhotometricInterpretation'):
            if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                # Invert grayscale values for MONOCHROME1
                pixel_array = np.max(pixel_array) - pixel_array
        
        # Apply window/level if present
        if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
            # Handle MultiValue objects by taking the first value
            window_center = dicom_data.WindowCenter
            window_width = dicom_data.WindowWidth
            
            # Convert to float, handling MultiValue cases
            if hasattr(window_center, '__iter__') and not isinstance(window_center, str):
                window_center = float(window_center[0])
            else:
                window_center = float(window_center)
                
            if hasattr(window_width, '__iter__') and not isinstance(window_width, str):
                window_width = float(window_width[0])
            else:
                window_width = float(window_width)
            
            # Apply windowing
            img_min = window_center - window_width // 2
            img_max = window_center + window_width // 2
            pixel_array = np.clip(pixel_array, img_min, img_max)
        
        # Normalize to 0-255 range
        pixel_array = pixel_array.astype(np.float64)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
        pixel_array = (pixel_array * 255).astype(np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(pixel_array)
        
        return image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing DICOM file: {str(e)}")


def process_standard_image(contents: bytes) -> Image.Image:
    """
    Process standard image formats and convert to PIL Image.
    
    Args:
        contents: Raw bytes of the image file
    
    Returns:
        PIL Image object
    """
    try:
        image = Image.open(io.BytesIO(contents))
        return image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image file: {str(e)}")


# Optional: Add a separate endpoint specifically for DICOM metadata extraction
@router.post("/dicom-info")
async def get_dicom_info(file: UploadFile = File(...)):
    """
    Extract and return DICOM metadata information.
    
    Args:
        file: The uploaded DICOM file
    
    Returns:
        Dictionary containing DICOM metadata
    """
    try:
        if not (file.filename.lower().endswith('.dcm') or file.content_type == 'application/dicom'):
            raise HTTPException(status_code=400, detail="File must be a DICOM file (.dcm)")
        
        contents = await file.read()
        dicom_data = pydicom.dcmread(io.BytesIO(contents))
        
        # Extract relevant metadata
        metadata = {
            "patient_id": getattr(dicom_data, 'PatientID', 'N/A'),
            "study_date": getattr(dicom_data, 'StudyDate', 'N/A'),
            "modality": getattr(dicom_data, 'Modality', 'N/A'),
            "manufacturer": getattr(dicom_data, 'Manufacturer', 'N/A'),
            "image_dimensions": f"{dicom_data.Rows}x{dicom_data.Columns}",
            "photometric_interpretation": getattr(dicom_data, 'PhotometricInterpretation', 'N/A'),
            "bits_allocated": getattr(dicom_data, 'BitsAllocated', 'N/A'),
            "window_center": getattr(dicom_data, 'WindowCenter', 'N/A'),
            "window_width": getattr(dicom_data, 'WindowWidth', 'N/A'),
        }
        
        return metadata
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing DICOM file: {str(e)}")



import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import io
from PIL import Image
import numpy as np
from fastapi import HTTPException, Query
from fastapi.responses import Response
import base64

# Add this to your existing router in images.py

@router.post("/segment-from-aws-path")
async def segment_from_aws_path(
    aws_path: str = Query(..., description="AWS S3 path to the image file"),
    return_format: str = Query("image", description="Return format: 'image' for PNG response or 'base64' for base64 string")
):
    """
    Endpoint that takes an AWS S3 path as input, downloads the image, 
    performs segmentation, and returns the segmentation mask.
    
    Args:
        aws_path: The S3 path to the image (e.g., "bucket-name/folder/image.png" or full S3 URL)
        return_format: "image" to return PNG response, "base64" to return base64 encoded string
    
    Returns:
        The segmentation mask as a PNG image or base64 string
    """
    try:
        # Download image from AWS S3
        image = download_image_from_s3(aws_path)
        
        # Perform segmentation
        mask_image = perform_segmentation(image)
        
        if return_format.lower() == "base64":
            # Convert mask to base64
            mask_byte_arr = io.BytesIO()
            mask_image.save(mask_byte_arr, format='PNG')
            mask_base64 = base64.b64encode(mask_byte_arr.getvalue()).decode('utf-8')
            
            return {
                "success": True,
                "mask_base64": mask_base64,
                "aws_path": aws_path,
                "format": "base64"
            }
        else:
            # Return mask as PNG image
            img_byte_arr = io.BytesIO()
            mask_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image from AWS path: {str(e)}")


def download_image_from_s3(aws_path: str) -> Image.Image:
    """
    Download image from AWS S3 using the provided path.
    
    Args:
        aws_path: S3 path - can be "bucket/key" format or full S3 URL
    
    Returns:
        PIL Image object
    """
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Parse the AWS path
        bucket_name, object_key = parse_s3_path(aws_path)
        
        # Download the image from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        image_content = response['Body'].read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_content))
        
        return image
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            raise HTTPException(status_code=404, detail=f"Image not found at path: {aws_path}")
        elif error_code == 'NoSuchBucket':
            raise HTTPException(status_code=404, detail=f"Bucket not found in path: {aws_path}")
        else:
            raise HTTPException(status_code=500, detail=f"AWS S3 error: {str(e)}")
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not configured")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading image from S3: {str(e)}")


def parse_s3_path(aws_path: str) -> tuple:
    """
    Parse S3 path to extract bucket name and object key.
    
    Args:
        aws_path: S3 path in various formats:
                 - "bucket-name/folder/image.png"
                 - "s3://bucket-name/folder/image.png"
                 - "https://bucket-name.s3.amazonaws.com/folder/image.png"
                 - "https://s3.amazonaws.com/bucket-name/folder/image.png"
    
    Returns:
        Tuple of (bucket_name, object_key)
    """
    try:
        # Remove leading/trailing whitespace
        aws_path = aws_path.strip()
        
        # Handle different S3 URL formats
        if aws_path.startswith('s3://'):
            # Format: s3://bucket-name/folder/image.png
            path_parts = aws_path[5:].split('/', 1)
            bucket_name = path_parts[0]
            object_key = path_parts[1] if len(path_parts) > 1 else ''
            
        elif aws_path.startswith('https://') and '.s3.amazonaws.com' in aws_path:
            # Format: https://bucket-name.s3.amazonaws.com/folder/image.png
            url_parts = aws_path.replace('https://', '').split('.s3.amazonaws.com/', 1)
            bucket_name = url_parts[0]
            object_key = url_parts[1] if len(url_parts) > 1 else ''
            
        elif aws_path.startswith('https://s3.amazonaws.com/'):
            # Format: https://s3.amazonaws.com/bucket-name/folder/image.png
            path_parts = aws_path.replace('https://s3.amazonaws.com/', '').split('/', 1)
            bucket_name = path_parts[0]
            object_key = path_parts[1] if len(path_parts) > 1 else ''
            
        else:
            # Assume format: bucket-name/folder/image.png
            path_parts = aws_path.split('/', 1)
            bucket_name = path_parts[0]
            object_key = path_parts[1] if len(path_parts) > 1 else ''
        
        if not bucket_name:
            raise ValueError("Bucket name not found in path")
        if not object_key:
            raise ValueError("Object key not found in path")
            
        return bucket_name, object_key
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid S3 path format: {aws_path}. Error: {str(e)}")


def perform_segmentation(image: Image.Image) -> Image.Image:
    """
    Perform segmentation on the input image using the loaded model.
    
    Args:
        image: PIL Image object
    
    Returns:
        PIL Image object containing the segmentation mask
    """
    try:
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to match model input
        image = image.resize((256, 256))
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=[0, -1])
        
        # Generate prediction using the model
        prediction = model2.predict(img_array)
        
        # Post-process the prediction
        mask = post_process_predictions(prediction[0, :, :, 0])
        
        # Convert mask to PIL Image
        mask_image = Image.fromarray(mask.astype(np.uint8))
        
        return mask_image
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")


# Alternative endpoint that uses presigned URLs (if you prefer this approach)
@router.post("/segment-from-presigned-url")
async def segment_from_presigned_url(
    presigned_url: str = Query(..., description="Presigned URL to the image file"),
    return_format: str = Query("image", description="Return format: 'image' for PNG response or 'base64' for base64 string")
):
    """
    Endpoint that takes a presigned URL as input, downloads the image, 
    performs segmentation, and returns the segmentation mask.
    
    Args:
        presigned_url: The presigned URL to the image
        return_format: "image" to return PNG response, "base64" to return base64 encoded string
    
    Returns:
        The segmentation mask as a PNG image or base64 string
    """
    try:
        # Download image from presigned URL
        response = requests.get(presigned_url)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        
        # Perform segmentation
        mask_image = perform_segmentation(image)
        
        if return_format.lower() == "base64":
            # Convert mask to base64
            mask_byte_arr = io.BytesIO()
            mask_image.save(mask_byte_arr, format='PNG')
            mask_base64 = base64.b64encode(mask_byte_arr.getvalue()).decode('utf-8')
            
            return {
                "success": True,
                "mask_base64": mask_base64,
                "presigned_url": presigned_url,
                "format": "base64"
            }
        else:
            # Return mask as PNG image
            img_byte_arr = io.BytesIO()
            mask_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image from URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image from presigned URL: {str(e)}")
    
    
    
class ImageRequest(BaseModel):
    image_url: str

@router.post("/api/get-image-for-pdf")
async def get_image_for_pdf(request: ImageRequest):
    """
    Proxy endpoint to fetch images from AWS S3 for PDF generation
    This bypasses CORS issues by fetching the image server-side
    """
    try:
        # Fetch the image from AWS S3
        response = requests.get(request.image_url, timeout=30)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {response.status_code}")
        
        # Get content type from the response
        content_type = response.headers.get('content-type', 'image/jpeg')
        
        # Return the image as streaming response
        return StreamingResponse(
            io.BytesIO(response.content),
            media_type=content_type,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")



