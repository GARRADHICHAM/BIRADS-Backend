# app/routers/annotations.py
from fastapi import APIRouter, Depends, HTTPException , Query, Path
from schemas.models import User

from database import images_collection, annotations_collection
from utils.s3 import upload_to_s3
from datetime import datetime
from bson.objectid import ObjectId
import base64
from fastapi import Body
import io
from typing import List, Optional
from datetime import datetime, timedelta
from pymongo import MongoClient, DESCENDING


router = APIRouter()

from fastapi import  Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

import base64
import os


router = APIRouter()




# Pydantic model for annotation data validation
class AnnotationCreate(BaseModel):
    image_id: str
    image_path: str
    file_name: str
    folder_name: Optional[str] = None
    annotation_base64: str
    birads_category: str
    lesion_type: str
    annotated_by: str
    annotator_name: Optional[str] = None
    notes: Optional[str] = None
    annotation_date: Optional[datetime] = Field(default_factory=datetime.now)
    dice_score: Optional[float] = None


# Helper function to convert string ID to ObjectId
def parse_object_id(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except:
        return None

# Route to save annotation
@router.post("/api/mammogram/save_annotation", status_code=201)
async def save_annotation(annotation: AnnotationCreate = Body(...)):
    try:
        # Convert image_id to ObjectId if it's in the proper format
        image_obj_id = parse_object_id(annotation.image_id)
        
        # Create annotation document
        annotation_doc = annotation.dict()
        
        # Replace string ID with ObjectId if valid
        if image_obj_id:
            annotation_doc["image_id"] = image_obj_id
            
        # Add timestamps
        annotation_doc["created_at"] = datetime.now()
        annotation_doc["updated_at"] = datetime.now()
        
        
        # Insert into MongoDB
        result = annotations_collection.insert_one(annotation_doc)
        
        
        return {"message": "Annotation saved successfully", "annotation_id": str(result.inserted_id)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving annotation: {str(e)}")
    
    
    
@router.get("/api/annotations/user/{user_id}", response_model=List[dict])
async def get_user_annotation_history(
    user_id: str = Path(..., description="User ID to retrieve annotations for"),
    limit: int = Query(20, description="Number of annotations to return"),
    skip: int = Query(0, description="Number of annotations to skip"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    birads_category: Optional[str] = Query(None, description="Filter by BI-RADS category"),
    lesion_type: Optional[str] = Query(None, description="Filter by lesion type")
):
    """Retrieve annotation history for a specific user"""
    try:
        # Build the query filter
        query = {"annotated_by": user_id}
        
        # Add date filters if provided
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter["$gte"] = start_date
            if end_date:
                date_filter["$lte"] = end_date
            query["annotation_date"] = date_filter
            
        # Add other filters
        if birads_category:
            query["birads_category"] = birads_category
        if lesion_type:
            query["lesion_type"] = lesion_type
            
        # Perform the query
        cursor = annotations_collection.find(query).sort("annotation_date", DESCENDING).skip(skip).limit(limit)
        
        # Convert to list and format for JSON
        annotations = []
        for doc in cursor:
            # Convert ObjectId to string
            doc["_id"] = str(doc["_id"])
            doc["image_id"] = str(doc["image_id"]) if isinstance(doc.get("image_id"), ObjectId) else doc.get("image_id")
            
            # Get a small version of the annotation for preview
            doc["annotation_preview"] = doc.get("annotation_base64", "")[:100] + "..." if doc.get("annotation_base64") else None
            
            # Remove large fields to make response lighter
            if "annotation_base64" in doc:
                del doc["annotation_base64"]
            if "annotation_history" in doc:
                del doc["annotation_history"]
                
            annotations.append(doc)
            
        return annotations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving annotation history: {str(e)}")

@router.get("/api/annotations/stats/user/{user_id}")
async def get_user_annotation_stats(user_id: str):
    """Get statistics about user annotations"""
    try:
        # Total annotations
        total_count = annotations_collection.count_documents({"annotated_by": user_id})
        
        # Annotations by BI-RADS category
        pipeline = [
            {"$match": {"annotated_by": user_id}},
            {"$group": {"_id": "$birads_category", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        birads_stats = list(annotations_collection.aggregate(pipeline))
        
        # Annotations by lesion type
        pipeline = [
            {"$match": {"annotated_by": user_id}},
            {"$group": {"_id": "$lesion_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        lesion_stats = list(annotations_collection.aggregate(pipeline))
        
        # Recent activity (annotations per day for last 7 days)
        seven_days_ago = datetime.now() - timedelta(days=7)
        pipeline = [
            {"$match": {
                "annotated_by": user_id,
                "annotation_date": {"$gte": seven_days_ago}
            }},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$annotation_date"}},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        daily_activity = list(annotations_collection.aggregate(pipeline))
        
        return {
            "total_annotations": total_count,
            "birads_distribution": birads_stats,
            "lesion_distribution": lesion_stats,
            "daily_activity": daily_activity
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving annotation stats: {str(e)}")
    
    
@router.get("/api/annotations/{annotation_id}", response_model=dict)
async def get_annotation_by_id(
    annotation_id: str = Path(..., description="Annotation ID to retrieve")
):
    """Retrieve a specific annotation by its ID"""
    try:
        # Convert string ID to ObjectId
        obj_id = parse_object_id(annotation_id)
        if not obj_id:
            raise HTTPException(status_code=400, detail="Invalid annotation ID format")
            
        # Query the database for the annotation
        annotation = annotations_collection.find_one({"_id": obj_id})
        
        # Check if annotation exists
        if not annotation:
            raise HTTPException(status_code=404, detail="Annotation not found")
            
        # Convert ObjectId to string for JSON serialization
        annotation["_id"] = str(annotation["_id"])
        annotation["image_id"] = str(annotation["image_id"]) if isinstance(annotation.get("image_id"), ObjectId) else annotation.get("image_id")
        
        return annotation
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving annotation: {str(e)}")
    
    
    # Pydantic model for annotation update validation
class AnnotationUpdate(BaseModel):
    image_id: Optional[str] = None
    image_path: Optional[str] = None
    file_name: Optional[str] = None
    folder_name: Optional[str] = None
    annotation_base64: Optional[str] = None
    birads_category: Optional[str] = None
    lesion_type: Optional[str] = None
    annotator_name: Optional[str] = None
    notes: Optional[str] = None
    dice_score: Optional[float] = None

@router.put("/api/updateAnnotations/{annotation_id}", response_model=dict)
async def update_annotation(
    annotation_id: str = Path(..., description="Annotation ID to update"),
    update_data: AnnotationUpdate = Body(..., description="Annotation update data")
):
    """Update an existing annotation by its ID"""
    try:
        # Convert string ID to ObjectId
        obj_id = parse_object_id(annotation_id)
        if not obj_id:
            raise HTTPException(status_code=400, detail="Invalid annotation ID format")
        
        # Prepare update document
        update_doc = {}
        # Only include non-None fields in the update
        update_fields = update_data.dict(exclude_unset=True)
        
        # Special handling for image_id to convert to ObjectId if needed
        if 'image_id' in update_fields:
            image_obj_id = parse_object_id(update_fields['image_id'])
            update_fields['image_id'] = image_obj_id if image_obj_id else update_fields['image_id']
        
        # Add updated_at timestamp
        update_fields['updated_at'] = datetime.now()
        
        # Perform the update
        result = annotations_collection.update_one(
            {"_id": obj_id}, 
            {"$set": update_fields}
        )
        
        # Check if the annotation was found and updated
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Annotation not found or no changes made")
        
        # Retrieve the updated annotation
        updated_annotation = annotations_collection.find_one({"_id": obj_id})
        
        # Convert ObjectId to string for JSON serialization
        updated_annotation["_id"] = str(updated_annotation["_id"])
        updated_annotation["image_id"] = str(updated_annotation["image_id"]) if isinstance(updated_annotation.get("image_id"), ObjectId) else updated_annotation.get("image_id")
        
        return {
            "message": "Annotation updated successfully", 
            "annotation": updated_annotation
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating annotation: {str(e)}")
    
    
@router.get("/api/annotations/stats/image/{image_id}")
async def get_image_annotation_stats(image_id: str = Path(..., description="Image ID to retrieve stats for")):
    """Get statistics about annotations for a specific image"""
    try:
        # Convert string ID to ObjectId
        image_obj_id = parse_object_id(image_id)
        if not image_obj_id:
            raise HTTPException(status_code=400, detail="Invalid image ID format")
            
        # Check if image exists
        image = images_collection.find_one({"_id": image_obj_id})
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Count total annotations for this image
        total_annotations = annotations_collection.count_documents({"image_id": image_obj_id})
        
        # Count unique users who annotated this image
        unique_users = len(annotations_collection.distinct("annotated_by", {"image_id": image_obj_id}))
        
        # Get distribution of BIRADS categories
        birads_pipeline = [
            {"$match": {"image_id": image_obj_id}},
            {"$group": {
                "_id": "$birads_category",
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        birads_stats = list(annotations_collection.aggregate(birads_pipeline))
        
        # Calculate percentages for BIRADS categories
        birads_distribution = []
        for stat in birads_stats:
            category = stat["_id"]
            count = stat["count"]
            percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
            birads_distribution.append({
                "category": category,
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        # Get distribution of lesion types
        lesion_pipeline = [
            {"$match": {"image_id": image_obj_id}},
            {"$group": {
                "_id": "$lesion_type",
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}}
        ]
        lesion_stats = list(annotations_collection.aggregate(lesion_pipeline))
        
        # Calculate percentages for lesion types
        lesion_distribution = []
        for stat in lesion_stats:
            lesion = stat["_id"]
            count = stat["count"]
            percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
            lesion_distribution.append({
                "lesion_type": lesion,
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        # Get annotations by user
        user_pipeline = [
            {"$match": {"image_id": image_obj_id}},
            {"$group": {
                "_id": "$annotated_by",
                "count": {"$sum": 1},
                "annotator_name": {"$first": "$annotator_name"}
            }},
            {"$sort": {"count": -1}}
        ]
        user_stats = list(annotations_collection.aggregate(user_pipeline))
        
        # Format user stats with percentages
        user_distribution = []
        for stat in user_stats:
            user_id = stat["_id"]
            count = stat["count"]
            annotator_name = stat.get("annotator_name", "Unknown")
            percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
            user_distribution.append({
                "user_id": user_id,
                "annotator_name": annotator_name,
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        # Get consensus information (agreement between annotators)
        consensus_data = {}
        if unique_users > 1:
            # For BIRADS consensus
            most_common_birads = None
            highest_birads_count = 0
            for birads in birads_distribution:
                if birads["count"] > highest_birads_count:
                    highest_birads_count = birads["count"]
                    most_common_birads = birads["category"]
            
            birads_agreement = (highest_birads_count / total_annotations) * 100 if total_annotations > 0 else 0
            
            # For lesion type consensus
            most_common_lesion = None
            highest_lesion_count = 0
            for lesion in lesion_distribution:
                if lesion["count"] > highest_lesion_count:
                    highest_lesion_count = lesion["count"]
                    most_common_lesion = lesion["lesion_type"]
            
            lesion_agreement = (highest_lesion_count / total_annotations) * 100 if total_annotations > 0 else 0
            
            consensus_data = {
                "most_common_birads": most_common_birads,
                "birads_agreement_percentage": round(birads_agreement, 2),
                "most_common_lesion": most_common_lesion,
                "lesion_agreement_percentage": round(lesion_agreement, 2)
            }
        
        return {
            "image_id": image_id,
            "file_name": image.get("file_name", "Unknown"),
            "total_annotations": total_annotations,
            "unique_annotators": unique_users,
            "birads_distribution": birads_distribution,
            "lesion_distribution": lesion_distribution,
            "user_distribution": user_distribution,
            "consensus_data": consensus_data if unique_users > 1 else {"message": "Consensus analysis requires multiple annotators"}
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving image annotation stats: {str(e)}")
    
    
    
    #save upload iages annotation
    
# Updated AnnotationCreate model
class AnnotationCreateSingle(BaseModel):
    original_image: str  # base64 encoded original image
    annotation_mask: str  # base64 encoded canvas annotation (stored in MongoDB)
    file_name: str
    region: str
    birads_category: str
    lesion_type: str
    annotated_by: str
    annotator_name: Optional[str] = None
    notes: Optional[str] = None
    annotation_date: Optional[datetime] = Field(default_factory=datetime.now)
    dice_score: Optional[float] = None

# Helper function to map BI-RADS category to S3 folder
def get_s3_folder_from_birads(birads_category: str) -> str:
    """Map BI-RADS category to corresponding S3 bucket folder"""
    birads_to_folder = {
        '1': 'B1',
        '2': 'B2', 
        '3': 'B3',
        '4': 'B4',
        '5': 'B5',
        '6': 'B5',  # BI-RADS 6 goes to B5 folder
        '0': 'B1'   # BI-RADS 0 goes to B1 folder (default)
    }
    
    return birads_to_folder.get(birads_category, 'B1')  # Default to B1 if category not found
def base64_to_image_file(base64_string: str) -> io.BytesIO:
    """Convert base64 string to image file-like object for S3 upload"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image_file = io.BytesIO(image_data)
        
        return image_file
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

# Single endpoint to save annotation
@router.post("/api/mammogram/save_upload", status_code=201)
async def save_annotation(annotation: AnnotationCreateSingle = Body(...)):
    try:
        # Convert original image base64 to file object for S3 upload
        original_image_file = base64_to_image_file(annotation.original_image)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"{timestamp}_original_{annotation.file_name}"
        
        # Determine S3 folder based on BI-RADS category
        s3_folder = get_s3_folder_from_birads(annotation.birads_category)
        
        # Upload original image to S3 in the appropriate BI-RADS folder
      
        original_s3_path = upload_to_s3(
        original_image_file, 
        original_filename, 
        s3_folder
        )
        
        # Save image document to images collection
        image_doc = {
            "file_name": annotation.file_name,
            "s3_path": original_s3_path,
            "s3_folder": s3_folder,  # Store which folder it was uploaded to
            "birads_category": annotation.birads_category,  # Store BI-RADS category
            "uploaded_at": datetime.now(),
            "uploaded_by": annotation.annotated_by,
            "image_type": "mammogram"
        }
        
        image_result = images_collection.insert_one(image_doc)
        image_id = image_result.inserted_id
        
        # Create annotation document with mask stored as base64 in MongoDB
        annotation_doc = {
            "image_id": image_id,
            "image_path": original_s3_path,  # S3 path for original image
            "s3_folder": s3_folder,  # Store which S3 folder was selected
            "annotation_base64": annotation.annotation_mask,  # Canvas mask stored in MongoDB
            "file_name": annotation.file_name,
            "region": annotation.region,
            "birads_category": annotation.birads_category,
            "lesion_type": annotation.lesion_type,
            "annotated_by": annotation.annotated_by,
            "annotator_name": annotation.annotator_name,
            "notes": annotation.notes,
            "annotation_date": annotation.annotation_date,
            "dice_score": annotation.dice_score,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        # Insert annotation into MongoDB
        result = annotations_collection.insert_one(annotation_doc)
        
        return {
            "message": "Annotation saved successfully",
            "annotation_id": str(result.inserted_id),
            "image_id": str(image_id),
            "original_image_path": original_s3_path,
            "s3_folder": s3_folder,
            "birads_category": annotation.birads_category
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving annotation: {str(e)}")

