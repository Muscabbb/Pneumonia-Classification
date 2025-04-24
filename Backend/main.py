from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from PIL import Image, UnidentifiedImageError
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ---------------------------
# Model Definition (Exact Match)
# ---------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32 * 18 * 18, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ---------------------------
# FastAPI Initialization
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set to specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Device Setup & Load Model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
model.eval()

# ---------------------------
# Image Transform (for Test)
# ---------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

# ---------------------------
# Response Models
# ---------------------------
class PredictionResult(BaseModel):
    filename: str
    prediction: str
    confidence: float

# ---------------------------
# In-Memory Prediction History
# ---------------------------
history: List[PredictionResult] = []

# ---------------------------
# Predict Endpoint
# ---------------------------
@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item()
        prediction = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    result = PredictionResult(
        filename=file.filename,
        prediction=prediction,
        confidence=round(prob, 4)
    )

    history.append(result)
    return result

# ---------------------------
# History Endpoint
# ---------------------------
@app.get("/history", response_model=List[PredictionResult])
def get_history():
    return history