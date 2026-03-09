import io
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from inference import analyze_for_api

app = FastAPI(title="Seaweed Health Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    suffix = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {suffix}. Please upload a valid image file."
        )

    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = UPLOAD_DIR / temp_name

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Validate actual file content as image
        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        temp_path.write_bytes(contents)

        result = analyze_for_api(str(temp_path))
        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
