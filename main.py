# app/main.py
from fastapi import FastAPI, Depends
from routers import auth, images, annotations ,admin
from fastapi.middleware.cors import CORSMiddleware
from routers.auth import get_current_active_user

app = FastAPI(title="Mammogram Segmentation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Next.js app domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(images.router, prefix="/api", tags=["Images"])
app.include_router(annotations.router, prefix="/api", tags=["Annotations"])
app.include_router(admin.router, prefix="/api", tags=["admin"])

@app.get("/hello")
async def hello():
    return "Hello world, I'm Hicham."

@app.get("/protected-hello")
def protected_hello(current_user = Depends(get_current_active_user)):
    return f"Hello {current_user.username}, this is a protected endpoint."

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)