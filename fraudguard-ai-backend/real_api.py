#!/usr/bin/env python3
"""
Simplified ML API with real model - working version
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image
import io
import base64
import time
import uvicorn

app = FastAPI(title="FraudGuard AI - Real ML API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None
transform = None

class FinalModelDetector(nn.Module):
    """Real model class"""

    def __init__(self, num_classes=2):
        super(FinalModelDetector, self).__init__()

        self.backbone = timm.create_model('efficientnet_b1', pretrained=True, num_classes=0)
        self.num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        with torch.no_grad():
            self.classifier[-1].bias[0] = 1.0
            self.classifier[-1].bias[1] = -0.5

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def load_model():
    """Load the real trained model"""
    global model, device, transform

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Using device: {device}")

        transform = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model_path = "final_model.pth"
        print(f"📁 Loading model from: {model_path}")

        model = FinalModelDetector(num_classes=2)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        print("✅ Real model loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        model = None
        return False

def predict_fraud(image: Image.Image):
    """Make prediction with real model"""
    try:
        start_time = time.time()

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100

            # Fixed class mapping: 0=FRAUD, 1=NON-FRAUD (corrected based on training)
            fraud_prob = probabilities[0][0].item() * 100
            non_fraud_prob = probabilities[0][1].item() * 100

        processing_time = time.time() - start_time

        # Corrected class mapping: 0=FRAUD, 1=NON-FRAUD
        prediction = "FRAUD" if predicted_class == 0 else "NON-FRAUD"

        # Risk level logic
        if prediction == "FRAUD":
            if confidence >= 90: risk_level = "VERY HIGH"
            elif confidence >= 80: risk_level = "HIGH"
            elif confidence >= 65: risk_level = "MODERATE"
            else: risk_level = "LOW-MODERATE"
        else:
            if confidence >= 80: risk_level = "VERY LOW"
            elif confidence >= 65: risk_level = "LOW"
            else: risk_level = "UNCERTAIN"

        # Recommended action
        if prediction == "FRAUD":
            if confidence >= 90: action = "Immediate investigation required"
            elif confidence >= 80: action = "Priority investigation"
            elif confidence >= 65: action = "Standard review process"
            else: action = "Basic documentation review"
        else:
            if confidence >= 80: action = "Auto-approve claim"
            elif confidence >= 65: action = "Standard processing"
            else: action = "Additional verification"

        return {
            "prediction": prediction,
            "confidence": f"{confidence:.1f}",
            "fraudProbability": f"{fraud_prob:.1f}",
            "nonFraudProbability": f"{non_fraud_prob:.1f}",
            "riskLevel": risk_level,
            "recommendedAction": action,
            "processingTime": f"{processing_time:.2f}s"
        }

    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "FraudGuard AI - Real ML API",
        "model_loaded": model is not None,
        "status": "running"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.post("/predict-base64")
async def predict_base64(data: dict):
    """Real prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image_data = base64.b64decode(data.get('imageData', ''))
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        result = predict_fraud(image)

        return {
            "success": True,
            "prediction": result,
            "timestamp": time.time()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Load model on startup
print("🚀 Starting FraudGuard AI - Real ML API...")
model_loaded = load_model()

if __name__ == "__main__":
    import os

    if model_loaded:
        print("✅ Starting server with real model...")
        port = int(os.environ.get("PORT", 8000))
        host = "0.0.0.0"

        print(f"🌐 Server will run on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    else:
        print("❌ Model not loaded - cannot start server")
