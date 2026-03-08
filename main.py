from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid

app = FastAPI(title="Seaweed Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def root():
    return {"message": "Seaweed Detection API is running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    temp_name = f"{uuid.uuid4().hex}_{file.filename}"
    temp_path = UPLOAD_DIR / temp_name

    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # TODO:
    # 1. load image
    # 2. run segmentation
    # 3. run ConvNeXt-Tiny prediction
    # 4. generate Grad-CAM
    # 5. generate DeepSeek explanation
    # 6. return JSON result

    return {
        "prediction": "unhealthy",
        "confidence": 0.548,
        "healthy_probability": 0.452,
        "unhealthy_probability": 0.548,
        "explanation": "Placeholder response from backend."
    }