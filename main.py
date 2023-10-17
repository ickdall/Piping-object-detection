from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, Depends, Form, UploadFile, File
from PIL import Image
import cv2
import numpy as np

import torch
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요에 따라 원래 도메인을 지정하세요.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_best.pt')
# model.conf = 0.8  # confidence threshold (0-1)
# model.iou = 0.65  # NMS IoU threshold (0-1)
# model.classes = [0]  # class threshold = 0(person)

def process_frame(frame):
    pil_image = Image.fromarray(frame)
    results = model(pil_image, size=640)
    results.render()
    
    result_image = np.array(pil_image)

    xyxy = results.xyxy[0]
    for bbox in xyxy:
        x1, y1, x2, y2, confidence, class_id = bbox
        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    return result_image

@app.get("/")
#async def root():
def root():
    return {"message": "Hello"}

@app.get("/home")
def home():
    return {"message": "Hello_home"}

@app.post("/detect/")
async def detect(data: dict):
    print(data)
    video_url = data.get("filePath")
    video_url = f'C:/Users/user/Downloads/Project_Everytime (2){video_url}'
    video_name = data.get("VidName")
    
    print(video_url)
    output_video_name = video_name[:-4] + "_output.mp4"

    cap = cv2.VideoCapture(video_url)
    print('video load success')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "C:/Users/huns1/OneDrive/바탕 화면/Project_Everytime/upload/analysis/" + output_video_name
    output_stream = cv2.VideoWriter(output_path , fourcc, fps, (width, height))

    print('model start')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        output_stream.write(processed_frame)
    print('model end')

    cap.release()
    output_stream.release()
    return_path = f"upload/analysis/{output_video_name}"
    print(output_path)

    return {"result_video_url": return_path}