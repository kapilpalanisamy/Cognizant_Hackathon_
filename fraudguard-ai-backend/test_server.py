#!/usr/bin/env python3
"""
Simple test server for ML API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Test ML API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Test ML API is running", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "test-ml-api"}

@app.post("/predict-base64")
async def predict_test(data: dict):
    """Mock prediction endpoint with realistic variety"""
    import random
    import hashlib
    
    # Use image data hash to get consistent results for same image
    image_data = data.get('imageData', '')
    image_hash = hashlib.md5(image_data.encode()).hexdigest()
    random.seed(int(image_hash[:8], 16))
    
    # Generate realistic predictions (60% non-fraud, 40% fraud like real data)
    is_fraud = random.random() < 0.4
    
    if is_fraud:
        prediction = "FRAUD"
        confidence = random.uniform(75, 95)
        fraud_prob = confidence
        risk_levels = ["HIGH", "VERY HIGH", "MODERATE"]
        risk_level = random.choice(risk_levels)
        actions = [
            "Priority investigation required",
            "Immediate investigation required", 
            "Standard review process"
        ]
    else:
        prediction = "NON-FRAUD"
        confidence = random.uniform(70, 90)
        fraud_prob = 100 - confidence
        risk_levels = ["LOW", "VERY LOW"]
        risk_level = random.choice(risk_levels)
        actions = [
            "Auto-approve claim",
            "Standard processing",
            "Basic documentation review"
        ]
    
    return {
        "success": True,
        "prediction": {
            "prediction": prediction,
            "confidence": f"{confidence:.1f}",
            "fraudProbability": f"{fraud_prob:.1f}",
            "nonFraudProbability": f"{100 - fraud_prob:.1f}",
            "riskLevel": risk_level,
            "recommendedAction": random.choice(actions),
            "processingTime": f"{random.uniform(0.8, 2.5):.1f}s"
        }
    }

if __name__ == "__main__":
    print("🧪 Starting Test ML API Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)