# app/schemas/models.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class User(BaseModel):
    username: str
    email: str
    full_name: str
    disabled: Optional[bool] = False

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class ImageMetadata(BaseModel):
    original_path: str
    mask_path: Optional[str] = None
    birads_score: Optional[int] = None
    lesion_type: Optional[str] = None
    uploaded_by: str
    uploaded_at: datetime
    status: str  # "pending", "annotated", "confirmed"