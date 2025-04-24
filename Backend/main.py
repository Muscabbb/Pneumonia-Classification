from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import uvicorn
import os

# Your custom model class (same as training)
class PneumoniaCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256 * 14 * 14, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for your frontend (adjust origin if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN()
model.load_state_dict(torch.load("pneumonia_cnn.pth", map_location=device))
model.eval().to(device)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# Class labels
class_names = ["NORMAL", "PNEUMONIA"]

# In-memory prediction history
history = []

class PredictionResult(BaseModel):
    filename: str
    prediction: str
    confidence: float

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    result = {
        "filename": file.filename,
        "prediction": class_names[pred.item()],
        "confidence": round(conf.item(), 4)
    }
    history.append(result)
    return result

@app.get("/history", response_model=List[PredictionResult])
def get_history():
    return history