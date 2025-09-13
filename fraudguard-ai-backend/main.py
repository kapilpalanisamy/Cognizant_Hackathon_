from fastapi import FastAPI, UploadFile, File, HTTPException
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
import json
from typing import Dict, Any

app = FastAPI(title="FraudGuard AI - ML API", version="1.0.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
device = None
transform = None

class FinalModelDetector(nn.Module):
    """
    ⚡ Fast precision detector - optimized EfficientNet-B1 for speed
    """
    
    def __init__(self, num_classes=2):
        super(FinalModelDetector, self).__init__()
        
        # EfficientNet-B1 - good balance of speed and accuracy
        self.backbone = timm.create_model('efficientnet_b1', pretrained=True, num_classes=0)
        self.num_features = self.backbone.num_features
        
        # Streamlined classifier for speed
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
        
        # Precision-optimized bias initialization
        with torch.no_grad():
            self.classifier[-1].bias[0] = 1.0   # Moderate bias toward fraud
            self.classifier[-1].bias[1] = -0.5  # Moderate bias against non-fraud
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def load_model():
    """Load the trained PyTorch model"""
    global model, device, transform
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Using device: {device}")
    
    # Setup image preprocessing
    transform = transforms.Compose([
        transforms.Resize((240, 240)),  # Match training size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        model_path = "final_model.pth"
        print(f"📁 Loading model from: {model_path}")
        
        # Create model instance first
        model = FinalModelDetector(num_classes=2)
        
        # Load state dict directly
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully!")
        print("📊 Model Specifications:")
        print("   - Architecture: EfficientNet-B1 based")
        print("   - Training Performance: 87.9% Precision, 86.0% Recall")
        print("   - Accuracy: 91.4%")
        print("   ✅ PRODUCTION READY!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print(f"📋 Debug info: {type(e).__name__}")
        model = None
        return False

def predict_fraud(image: Image.Image) -> Dict[str, Any]:
    """
    Predict fraud for a single image using the trained model
    """
    try:
        start_time = time.time()
        
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        # Class names
        class_names = ['FRAUD', 'NON-FRAUD']
        prediction = class_names[predicted_class]
        fraud_prob = probabilities[0][0].item() * 100
        non_fraud_prob = probabilities[0][1].item() * 100
        
        # Determine risk level
        if prediction == 'FRAUD':
            if confidence >= 90:
                risk_level = 'VERY HIGH'
                recommended_action = 'Immediate investigation required - Auto-reject consideration'
            elif confidence >= 80:
                risk_level = 'HIGH'
                recommended_action = 'Priority investigation - Assign to senior investigator'
            elif confidence >= 65:
                risk_level = 'MODERATE'
                recommended_action = 'Standard review process - Regular investigation queue'
            else:
                risk_level = 'LOW-MODERATE'
                recommended_action = 'Basic documentation review - Low priority'
        else:  # NON-FRAUD
            if confidence >= 80:
                risk_level = 'VERY LOW'
                recommended_action = 'Auto-approve claim - Fast-track payment'
            elif confidence >= 65:
                risk_level = 'LOW'
                recommended_action = 'Standard processing - Normal workflow'
            else:
                risk_level = 'UNCERTAIN'
                recommended_action = 'Additional verification - Manual review recommended'
        
        processing_time = time.time() - start_time
        
        return {
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'fraudProbability': round(fraud_prob, 1),
            'nonFraudProbability': round(non_fraud_prob, 1),
            'riskLevel': risk_level,
            'recommendedAction': recommended_action,
            'processingTime': f"{processing_time:.2f}s",
            'modelInfo': {
                'architecture': 'EfficientNet-B1 FinalModelDetector',
                'precision': '87.9%',
                'recall': '86.0%',
                'accuracy': '91.4%'
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        print("⚠️  Model loading failed - API will return errors")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "FraudGuard AI - ML API",
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not_set"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not_loaded",
        "device": str(device) if device else "not_set",
        "gpu_available": torch.cuda.is_available(),
        "model_info": {
            "architecture": "EfficientNet-B1 FinalModelDetector",
            "precision": "87.9%",
            "recall": "86.0%", 
            "accuracy": "91.4%"
        }
    }

@app.post("/predict")
async def predict_fraud_endpoint(file: UploadFile = File(...)):
    """
    Predict fraud from uploaded image file
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Get prediction
        result = predict_fraud(image)
        
        return {
            "success": True,
            "filename": file.filename,
            "prediction": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/predict-base64")
async def predict_fraud_base64(data: dict):
    """
    Predict fraud from base64 encoded image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data.get('imageData', ''))
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Get prediction
        result = predict_fraud(image)
        
        return {
            "success": True,
            "prediction": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FraudGuard AI - ML API Server...")
    print("=" * 50)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        input("Press Enter to exit...")