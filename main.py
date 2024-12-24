import os
import sys
import shutil
import cv2
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from ultralytics import YOLO

app = FastAPI()


UPLOAD_DIR = "uploaded_videos"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise ValueError(f"Invalid FPS detected: {fps}")
    frame_interval = int(fps / frame_rate)
    frames = []
    timestamps = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
            timestamps.append(frame_count / fps)  # Timestamp in seconds
        frame_count += 1

    cap.release()
    return frames, timestamps

def predict_smoking(frames, model_path):
    model = YOLO(model_path)
    smoking_indices = []

    for i, frame in enumerate(frames):
        results = model(frame)  # Run prediction

        for result in results:
            if result.probs is not None: 
                class_index = result.probs.top1  # Get the index of the top-1 class
                class_name = result.names[class_index]  # Map index to class name
                if class_name == "smoking":  
                    smoking_indices.append(i)  
            elif result.boxes is not None:  
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    if class_name == "smoking":
                        smoking_indices.append(i)

    return smoking_indices


def map_indices_to_timestamps(smoking_indices, timestamps):
    return [timestamps[idx] for idx in smoking_indices]


@app.get("/", response_class=HTMLResponse)
async def render_upload_page(request: Request):
    """
    Renders the video upload page.
    """
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/detect-smoking/")
async def detect_smoking(video: UploadFile = File(...)):
    """
    Detects smoking activity in an uploaded video.
    """
    video_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        # Extract frames and timestamps
        frames, timestamps = extract_frames(video_path)

        # Run detection
        model_path = "C:/Users/a1410/Downloads/smoking/runs/classify/train/weights/best.pt"
        smoking_indices = predict_smoking(frames, model_path)

        # Map indices to timestamps
        smoking_timestamps = map_indices_to_timestamps(smoking_indices, timestamps)

        return {"smoking_timestamps": smoking_timestamps}

    except Exception as e:
        return {"error": str(e)}



