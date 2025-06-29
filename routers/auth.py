# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Optional
from database import get_database, get_users_collection
from models.user import UserCreate, UserInDB, User
from bson import ObjectId

# JWT settings
SECRET_KEY = "Garrad2225"  # Change this to a secure random key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 2880

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

router = APIRouter(prefix="/auth")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str):
    users_collection = get_users_collection()
    user = users_collection.find_one({"username": username})
    if user:
        # Convert ObjectId to string
        user["_id"] = str(user["_id"])
        return UserInDB(**user)
    return None

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@router.post("/register", response_model=User)
def register_user(user: UserCreate):
    users_collection = get_users_collection()
    
    # Check if username already exists
    existing_user = users_collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
        
    # Create new user
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    user_dict.pop("password")
    user_in_db = UserInDB(**user_dict, hashed_password=hashed_password, disabled=False)
    
    # Insert into database
    new_user_id = users_collection.insert_one(user_in_db.dict()).inserted_id
    created_user = users_collection.find_one({"_id": new_user_id})
    
    # Convert ObjectId to string for response
    created_user["_id"] = str(created_user["_id"])
    return User(**created_user)

@router.post("/login", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=User)
def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


# Add this model class after your existing models
class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None

# Add this endpoint after your existing endpoints
@router.put("/profile", response_model=User)
def update_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user)
):
    users_collection = get_users_collection()
    
    # Prepare update data
    update_data = {}
    
    if user_update.username is not None:
        # Check if username is already taken by another user
        existing_user = users_collection.find_one({
            "username": user_update.username,
            "_id": {"$ne": ObjectId(current_user.id)}
        })
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        update_data["username"] = user_update.username
    
    if user_update.full_name is not None:
        update_data["full_name"] = user_update.full_name
    
    if user_update.email is not None:
        # Check if email is already taken by another user
        existing_user = users_collection.find_one({
            "email": user_update.email,
            "_id": {"$ne": ObjectId(current_user.id)}
        })
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        update_data["email"] = user_update.email
    
    if user_update.password is not None:
        update_data["hashed_password"] = get_password_hash(user_update.password)
    
    # If no fields to update
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided for update"
        )
    
    # Update user in database
    result = users_collection.update_one(
        {"_id": ObjectId(current_user.id)},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Return updated user
    updated_user = users_collection.find_one({"_id": ObjectId(current_user.id)})
    updated_user["_id"] = str(updated_user["_id"])
    return User(**updated_user)

# Add this endpoint after your existing endpoints
@router.put("/users/{user_id}/toggle-status")
async def toggle_user_status(
    user_id: str,
    current_user: User = Depends(get_current_active_user)
):
    users_collection = get_users_collection()
    
    # Find the user
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Toggle the disabled status
    new_status = not user.get('disabled', False)
    
    # Update the user
    result = users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"disabled": new_status}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update user status"
        )
    
    return {"status": "success", "disabled": new_status}