from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from bson.objectid import ObjectId
from pymongo import MongoClient, DESCENDING, ASCENDING
from database import images_collection, annotations_collection, users_collection
import base64
import os
from collections import defaultdict

router = APIRouter()

# Pydantic models for responses
class DashboardStats(BaseModel):
    total_images: int
    total_annotations: int
    active_users: int
    model_accuracy: float
    dice_score: float
    pending_reviews: int

class BiradsDistribution(BaseModel):
    name: str
    value: int
    color: str

class PerformanceData(BaseModel):
    month: str
    accuracy: float
    dice: float
    sensitivity: float
    specificity: float

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str
    status: str
    disabled: bool
    annotations: int
    accuracy: float
    created_at: Optional[datetime] = None

class AnnotationResponse(BaseModel):
    id: str
    image_id: str
    image_path: str
    file_name: str
    annotator: str
    annotator_name: Optional[str] = None
    annotation_base64: str
    birads_category: str
    lesion_type: str
    status: str
    confidence: Optional[float] = None
    annotation_date: datetime
    dice_score: Optional[float] = None

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    disabled: Optional[bool] = None

class AnnotationStatusUpdate(BaseModel):
    status: str
    notes: Optional[str] = None

# Helper function to convert ObjectId to string
def serialize_doc(doc):
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

# Helper function to parse ObjectId
def parse_object_id(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except:
        return None

# Dashboard endpoints
@router.get("/api/admin/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    try:
        # Get total counts
        total_images = images_collection.count_documents({})
        total_annotations = annotations_collection.count_documents({})
        active_users = users_collection.count_documents({"disabled": False})
        pending_reviews = annotations_collection.count_documents({"status": {"$in": ["Pending", "Under Review"]}})
        
        # Calculate model accuracy (average dice score)
        pipeline = [
            {"$match": {"dice_score": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": None, "avg_dice": {"$avg": "$dice_score"}}}
        ]
        dice_result = list(annotations_collection.aggregate(pipeline))
        avg_dice_score = dice_result[0]["avg_dice"] if dice_result else 0.0
        
        # Mock model accuracy calculation (you can implement actual logic)
        model_accuracy = min(95.0, avg_dice_score * 100 if avg_dice_score else 90.0)
        
        return DashboardStats(
            total_images=total_images,
            total_annotations=total_annotations,
            active_users=active_users,
            model_accuracy=0.92,
            dice_score= 0.85,
            pending_reviews=pending_reviews
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard stats: {str(e)}")

@router.get("/api/admin/dashboard/birads-distribution", response_model=List[BiradsDistribution])
async def get_birads_distribution():
    try:
        pipeline = [
            {"$group": {"_id": "$birads_category", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        
        results = list(annotations_collection.aggregate(pipeline))
        
        # Color mapping for BI-RADS categories
        colors = {
            "1": "#10B981",  # Green
            "2": "#3B82F6",  # Blue
            "3": "#F59E0B",  # Yellow
            "4": "#EF4444",  # Red
            "5": "#7C2D12"   # Dark Red
        }
        
        distribution = []
        for result in results:
            birads = str(result["_id"])
            distribution.append(BiradsDistribution(
                name=f"BI-RADS {birads}",
                value=result["count"],
                color=colors.get(birads, "#6B7280")
            ))
        
        return distribution
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching BI-RADS distribution: {str(e)}")

@router.get("/api/admin/dashboard/performance-trends", response_model=List[PerformanceData])
async def get_performance_trends():
    try:
        # This is a mock implementation. In reality, you'd calculate these metrics
        # based on your actual annotation validation data
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        performance_data = []
        
        for i, month in enumerate(months):
            # Mock data with slight improvements over time
            base_accuracy = 91.2 + (i * 0.7)
            base_dice = 0.85 + (i * 0.007)
            base_sensitivity = 88.9 + (i * 0.7)
            base_specificity = 93.1 + (i * 0.5)
            
            performance_data.append(PerformanceData(
                month=month,
                accuracy=round(base_accuracy, 1),
                dice=round(base_dice, 2),
                sensitivity=round(base_sensitivity, 1),
                specificity=round(base_specificity, 1)
            ))
        
        return performance_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching performance trends: {str(e)}")

# User Management endpoints
@router.get("/api/admin/users", response_model=List[UserResponse])
async def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None, regex="^(Active|Inactive)$")
):
    try:
        # Build filter
        filter_dict = {}
        if status:
            filter_dict["disabled"] = (status == "Inactive")
        
        # Get users
        users_cursor = users_collection.find(filter_dict).skip(skip).limit(limit)
        users = []
        
        for user in users_cursor:
            # Count annotations for this user
            annotation_count = annotations_collection.count_documents({"annotated_by": str(user["_id"])})
            
            # Calculate accuracy (mock calculation based on dice scores)
            accuracy_pipeline = [
                {"$match": {"annotated_by": str(user["_id"]), "dice_score": {"$exists": True}}},
                {"$group": {"_id": None, "avg_accuracy": {"$avg": "$dice_score"}}}
            ]
            accuracy_result = list(annotations_collection.aggregate(accuracy_pipeline))
            accuracy = (accuracy_result[0]["avg_accuracy"] * 100) if accuracy_result else 90.0
            
            # Determine role (you might want to add a role field to your user model)
            role = "Radiologist" if "dr" in user.get("username", "").lower() else "Annotator"
            
            users.append(UserResponse(
                id=str(user["_id"]),
                name=user.get("full_name", "Unknown"),
                email=user.get("email", ""),
                role=role,
                status="Active" if not user.get("disabled", False) else "Inactive",
                disabled=user.get("disabled", False),
                annotations=annotation_count,
                accuracy=round(accuracy, 1),
                created_at=user.get("created_at")
            ))
        
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")

@router.put("/api/admin/users/{user_id}")
async def update_user(user_id: str, user_update: UserUpdate):
    try:
        user_obj_id = parse_object_id(user_id)
        if not user_obj_id:
            raise HTTPException(status_code=400, detail="Invalid user ID")
        
        update_dict = {k: v for k, v in user_update.dict().items() if v is not None}
        if not update_dict:
            raise HTTPException(status_code=400, detail="No update data provided")
        
        update_dict["updated_at"] = datetime.now()
        
        result = users_collection.update_one(
            {"_id": user_obj_id},
            {"$set": update_dict}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": "User updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating user: {str(e)}")

@router.get("/api/admin/annotations", response_model=List[AnnotationResponse])
async def get_annotations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    annotator: Optional[str] = Query(None),
    birads: Optional[str] = Query(None)
):
    try:
        # Build filter
        filter_dict = {}
        if status:
            filter_dict["status"] = status
        if annotator:
            filter_dict["annotated_by"] = annotator
        if birads:
            filter_dict["birads_category"] = birads
        
        # Get annotations with user details
        pipeline = [
            {"$match": filter_dict},
            {
                "$lookup": {
                    "from": "users",
                    "localField": "annotated_by",
                    "foreignField": "_id",
                    "as": "annotator_info"
                }
            },
            {"$skip": skip},
            {"$limit": limit},
            {"$sort": {"annotation_date": -1}}
        ]
        
        annotations = list(annotations_collection.aggregate(pipeline))
        
        result = []
        for ann in annotations:
            # Safely get annotator info - handle empty array case
            annotator_info_list = ann.get("annotator_info", [])
            annotator_info = annotator_info_list[0] if annotator_info_list else {}
            
            # Determine status if not set
            current_status = ann.get("status", "Pending")
            if not current_status:
                # Auto-determine status based on dice score or other criteria
                dice_score = ann.get("dice_score", 0)
                if dice_score and dice_score < 0.7:
                    current_status = "Flagged"
                elif dice_score and dice_score > 0.9:
                    current_status = "Approved"
                else:
                    current_status = "Under Review"
            
            result.append(AnnotationResponse(
                id=str(ann["_id"]),
                image_id=str(ann.get("image_id", "")),
                image_path=ann.get("image_path", ""),
                file_name=ann.get("file_name", ""),
                annotator=annotator_info.get("full_name", "Unknown"),
                annotator_name=ann.get("annotator_name", "Unknown"),
                annotation_base64 = ann.get("annotation_base64", ""),
                birads_category=ann.get("birads_category", ""),
                lesion_type=ann.get("lesion_type", ""),
                status=current_status,
                confidence=ann.get("dice_score"),
                annotation_date=ann.get("annotation_date", datetime.now()),
                dice_score=ann.get("dice_score")
            ))
        
        return result
    except Exception as e:
        logger.error(f"Error fetching annotations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching annotations: {str(e)}")

@router.put("/api/admin/annotations/{annotation_id}/status")
async def update_annotation_status(annotation_id: str, status_update: AnnotationStatusUpdate):
    try:
        annotation_obj_id = parse_object_id(annotation_id)
        if not annotation_obj_id:
            raise HTTPException(status_code=400, detail="Invalid annotation ID")
        
        update_dict = {
            "status": status_update.status,
            "updated_at": datetime.now()
        }
        
        if status_update.notes:
            update_dict["review_notes"] = status_update.notes
        
        result = annotations_collection.update_one(
            {"_id": annotation_obj_id},
            {"$set": update_dict}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        return {"message": "Annotation status updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating annotation status: {str(e)}")

@router.get("/api/admin/annotations/stats")
async def get_annotation_stats():
    try:
        pipeline = [
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        results = list(annotations_collection.aggregate(pipeline))
        
        stats = {
            "total": 0,
            "approved": 0,
            "pending": 0,
            "flagged": 0,
            "under_review": 0
        }
        
        for result in results:
            status = result["_id"] or "pending"
            count = result["count"]
            stats["total"] += count
            
            if status.lower() == "approved":
                stats["approved"] = count
            elif status.lower() == "pending":
                stats["pending"] = count
            elif status.lower() == "flagged":
                stats["flagged"] = count
            elif status.lower() in ["under review", "under_review"]:
                stats["under_review"] = count
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching annotation stats: {str(e)}")

# Quality Control endpoints
@router.get("/api/admin/quality/metrics")
async def get_quality_metrics():
    try:
        # Mock quality metrics - implement actual calculations based on your needs
        metrics = {
            "inter_observer_agreement": {
                "cohens_kappa": 0.82,
                "fleiss_kappa": 0.79,
                "pearson_correlation": 0.91
            },
            "quality_scores": {
                "average_quality": 8.7,
                "consistency_rate": 94.2,
                "error_rate": 2.1
            },
            "review_queue": {
                "pending_reviews": 23,
                "high_priority": 7,
                "avg_review_time": 2.3
            }
        }
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching quality metrics: {str(e)}")

@router.get("/api/admin/quality/flagged-annotations")
async def get_flagged_annotations():
    try:
        # Get annotations that need review
        filter_dict = {
            "$or": [
                {"status": "Flagged"},
                {"dice_score": {"$lt": 0.7}},
                {"status": {"$in": ["Under Review", "Pending"]}}
            ]
        }
        
        flagged = list(annotations_collection.find(filter_dict).limit(50).sort("annotation_date", -1))
        
        result = []
        for ann in flagged:
            # Determine issue type
            issue = "Under review"
            priority = "Medium"
            
            if ann.get("dice_score", 1.0) < 0.7:
                issue = "Low confidence score"
                priority = "High"
            elif ann.get("status") == "Flagged":
                issue = "Quality concern"
                priority = "High"
            
            result.append({
                "id": str(ann["_id"]),
                "image_id": ann.get("file_name", "Unknown"),
                "issue": issue,
                "annotator": ann.get("annotator_name", "Unknown"),
                "priority": priority,
                "date": ann.get("annotation_date", datetime.now()).strftime("%Y-%m-%d"),
                "confidence": ann.get("dice_score", 0.0)
            })
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching flagged annotations: {str(e)}")

# Export endpoints
@router.get("/api/admin/export/annotations")
async def export_annotations(
    format: str = Query("json", regex="^(json|csv)$"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    status: Optional[str] = Query(None)
):
    try:
        # Build filter
        filter_dict = {}
        if start_date and end_date:
            filter_dict["annotation_date"] = {"$gte": start_date, "$lte": end_date}
        if status:
            filter_dict["status"] = status
        
        annotations = list(annotations_collection.find(filter_dict))
        
        # Convert ObjectIds to strings
        for ann in annotations:
            ann["_id"] = str(ann["_id"])
            if "image_id" in ann and isinstance(ann["image_id"], ObjectId):
                ann["image_id"] = str(ann["image_id"])
        
        if format == "json":
            return {"data": annotations, "count": len(annotations)}
        else:
            # For CSV format, you might want to implement CSV conversion
            return {"message": "CSV export not implemented yet", "data": annotations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting annotations: {str(e)}")

# Additional utility endpoints
@router.delete("/api/admin/annotations/{annotation_id}")
async def delete_annotation(annotation_id: str):
    try:
        annotation_obj_id = parse_object_id(annotation_id)
        if not annotation_obj_id:
            raise HTTPException(status_code=400, detail="Invalid annotation ID")
        
        result = annotations_collection.delete_one({"_id": annotation_obj_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        return {"message": "Annotation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting annotation: {str(e)}")

@router.post("/api/admin/annotations/bulk-update")
async def bulk_update_annotations(
    annotation_ids: List[str] = Body(...),
    status: str = Body(...),
    notes: Optional[str] = Body(None)
):
    try:
        obj_ids = []
        for id_str in annotation_ids:
            obj_id = parse_object_id(id_str)
            if obj_id:
                obj_ids.append(obj_id)
        
        if not obj_ids:
            raise HTTPException(status_code=400, detail="No valid annotation IDs provided")
        
        update_dict = {
            "status": status,
            "updated_at": datetime.now()
        }
        
        if notes:
            update_dict["review_notes"] = notes
        
        result = annotations_collection.update_many(
            {"_id": {"$in": obj_ids}},
            {"$set": update_dict}
        )
        
        return {
            "message": f"Updated {result.modified_count} annotations successfully",
            "updated_count": result.modified_count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error bulk updating annotations: {str(e)}")