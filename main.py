import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import analyze_for_api

app = FastAPI(title="Seaweed Health Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "Seaweed Health Analyzer API is running",
        "status": "ok",
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = UPLOAD_DIR / temp_name

    try:
        contents = await file.read()
        temp_path.write_bytes(contents)

        result = analyze_for_api(str(temp_path))
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass