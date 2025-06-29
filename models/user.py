# app/models/user.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: Optional[str] = Field(None, alias="_id")
    disabled: bool = False
    created_at: Optional[datetime] = None
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "username": "johndoe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "disabled": False,
                "created_at": "2023-01-01T00:00:00"
            }
        }

class UserInDB(UserBase):
    id: Optional[str] = Field(None, alias="_id")
    hashed_password: str
    disabled: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)