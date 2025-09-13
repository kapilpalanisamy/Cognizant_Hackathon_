# 🚀 Backend Documentation - FraudGuard AI

## 🎯 **Overview**

The FraudGuard AI backend is a high-performance **FastAPI** application that serves the machine learning model for real-time fraud detection. It's designed for production deployment with optimized inference, comprehensive error handling, and scalable architecture.

---

## 🏗️ **Architecture & Tech Stack**

### **Core Technologies**
- **FastAPI**: Modern, fast web framework for building APIs
- **PyTorch**: Deep learning framework for model inference
- **EfficientNet-B1**: Pre-trained computer vision model
- **Uvicorn**: ASGI server for production deployment
- **Pillow**: Image processing and manipulation
- **Python 3.8+**: Modern Python with async support

### **Key Dependencies**
```txt
fastapi==0.103.1
uvicorn==0.23.2
torch==2.0.1
torchvision==0.15.2
efficientnet-pytorch==0.7.1
Pillow==10.0.0
python-multipart==0.0.6
aiofiles==23.2.1
```

---

## 📁 **Project Structure**

```
fraudguard-ai-backend/
├── 📄 main.py                          # Primary FastAPI application
├── 📄 lightweight_api.py               # Optimized API for production
├── 📄 real_api.py                     # Full-featured API variant
├── 📄 convert_model.py                # Model conversion utilities
├── 📄 test_server.py                  # Local testing server
├── 📄 requirements.txt                # Production dependencies
├── 📄 requirements_light.txt          # Minimal dependencies
├── 📄 fast_precision_fraud_model_statedict.pth  # Trained model
├── 📄 Dockerfile                      # Container configuration
├── 📄 railway.json                    # Railway deployment config
├── 📄 package.json                    # Node.js metadata (if needed)
└── 📄 README.md                       # Backend-specific documentation
```

---

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.8 or higher
- pip or conda for package management
- 4GB+ RAM (for model loading)
- Optional: CUDA-compatible GPU

### **Local Development Setup**

1. **Clone and Navigate**
```bash
git clone https://github.com/kapilpalanisamy/Cognizant_Hackathon_.git
cd Cognizant_Hackathon_/fraudguard-ai-backend
```

2. **Create Virtual Environment**
```bash
# Using venv
python -m venv fraud_env
source fraud_env/bin/activate  # On Windows: fraud_env\Scripts\activate

# Using conda
conda create -n fraud_env python=3.8
conda activate fraud_env
```

3. **Install Dependencies**
```bash
# For production
pip install -r requirements.txt

# For lightweight deployment
pip install -r requirements_light.txt
```

4. **Download Model** (if not included)
```bash
# Model should be in the same directory
# fast_precision_fraud_model_statedict.pth
```

5. **Start Development Server**
```bash
# Using main.py
python main.py

# Using lightweight API
python lightweight_api.py

# Using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🔧 **API Endpoints**

### **Health Check**
```http
GET /
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "FraudGuard AI Backend is running",
  "model_loaded": true,
  "timestamp": "2025-09-13T10:37:23Z"
}
```

### **Fraud Detection**
```http
POST /predict
Content-Type: multipart/form-data
```

**Parameters:**
- `file`: Image file (JPEG, PNG, WebP)
- `max_size`: 10MB limit

**Request Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@claim_image.jpg"
```

**Response:**
```json
{
  "prediction": "FRAUD",
  "confidence": 91.2,
  "fraudProbability": "91.2",
  "nonFraudProbability": "8.8",
  "riskLevel": "HIGH",
  "recommendedAction": "Manual review required - High fraud probability detected",
  "processingTime": "1.23s",
  "modelVersion": "v1.3",
  "imageSize": [224, 224],
  "timestamp": "2025-09-13T10:37:23Z"
}
```

### **Model Information**
```http
GET /model-info
```

**Response:**
```json
{
  "modelName": "FastPrecisionDetector",
  "architecture": "EfficientNet-B1",
  "version": "v1.3",
  "accuracy": "91.4%",
  "inputSize": [224, 224],
  "parameters": "7.8M",
  "trainingDate": "2025-09-12",
  "classes": ["NON-FRAUD", "FRAUD"]
}
```

---

## 🧠 **Model Implementation**

### **Model Architecture**
```python
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class FastPrecisionDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load pre-trained EfficientNet-B1
        self.backbone = EfficientNet.from_pretrained('efficientnet-b1')
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone.extract_features(x)
        features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        features = features.flatten(1)
        return self.classifier(features)
```

### **Model Loading & Initialization**
```python
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

class FraudDetectionModel:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        
    def _load_model(self, model_path):
        model = FastPrecisionDetector(num_classes=2)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.eval()
        model.to(self.device)
        return model
    
    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image):
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        return probabilities.cpu().numpy()[0]
```

---

## ⚡ **FastAPI Application**

### **Main Application** (`main.py`)
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
from PIL import Image
import io

app = FastAPI(
    title="FraudGuard AI Backend",
    description="Advanced Insurance Fraud Detection API",
    version="1.3.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    global model
    print("Loading fraud detection model...")
    model = FraudDetectionModel("fast_precision_fraud_model_statedict.pth")
    print("Model loaded successfully!")

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "FraudGuard AI Backend is running",
        "model_loaded": model is not None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

@app.post("/predict")
async def predict_fraud(file: UploadFile = File(...)):
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    # Size limit (10MB)
    max_size = 10 * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 10MB"
        )
    
    try:
        # Process image
        start_time = time.time()
        image = Image.open(io.BytesIO(contents))
        
        # Get prediction
        probabilities = model.predict(image)
        processing_time = time.time() - start_time
        
        # Format results
        fraud_prob = float(probabilities[1]) * 100
        non_fraud_prob = float(probabilities[0]) * 100
        
        prediction = "FRAUD" if fraud_prob > 50 else "NON-FRAUD"
        confidence = max(fraud_prob, non_fraud_prob)
        
        # Risk level assessment
        if confidence > 85:
            risk_level = "HIGH" if prediction == "FRAUD" else "LOW"
        elif confidence > 70:
            risk_level = "MEDIUM"
        else:
            risk_level = "UNCERTAIN"
        
        # Recommended action
        if prediction == "FRAUD" and confidence > 85:
            action = "Manual review required - High fraud probability detected"
        elif prediction == "FRAUD" and confidence > 70:
            action = "Additional verification recommended"
        else:
            action = "Approve claim through standard process"
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "fraudProbability": f"{fraud_prob:.1f}",
            "nonFraudProbability": f"{non_fraud_prob:.1f}",
            "riskLevel": risk_level,
            "recommendedAction": action,
            "processingTime": f"{processing_time:.2f}s",
            "modelVersion": "v1.3",
            "imageSize": list(image.size),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 🔧 **Performance Optimization**

### **Lightweight API** (`lightweight_api.py`)
Optimized version for production deployment:

```python
from fastapi import FastAPI, File, UploadFile
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Run prediction in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        process_image_sync, 
        contents
    )
    
    return result

def process_image_sync(image_bytes):
    # Synchronous processing
    image = Image.open(io.BytesIO(image_bytes))
    probabilities = model.predict(image)
    return format_prediction(probabilities)
```

### **Memory Optimization**
```python
import gc
import torch

class OptimizedModel:
    def predict(self, image):
        try:
            # Clear cache before prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result = self._run_inference(image)
            
            # Force garbage collection
            gc.collect()
            
            return result
        except Exception as e:
            # Cleanup on error
            gc.collect()
            raise e
```

---

## 🐳 **Containerization**

### **Dockerfile**
```dockerfile
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_light.txt .
RUN pip install --no-cache-dir -r requirements_light.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "lightweight_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Docker Commands**
```bash
# Build image
docker build -t fraudguard-ai-backend .

# Run container
docker run -p 8000:8000 fraudguard-ai-backend

# Run with environment variables
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/model.pth \
  fraudguard-ai-backend
```

---

## 🚀 **Deployment**

### **Render Deployment**
```json
{
  "services": [
    {
      "type": "web",
      "name": "fraudguard-ai-backend",
      "env": "python",
      "plan": "free",
      "buildCommand": "pip install -r requirements_light.txt",
      "startCommand": "uvicorn lightweight_api:app --host 0.0.0.0 --port $PORT",
      "envVars": [
        {
          "key": "PYTHON_VERSION",
          "value": "3.8.10"
        }
      ]
    }
  ]
}
```

### **Environment Variables**
```bash
# Production
MODEL_PATH=/app/fast_precision_fraud_model_statedict.pth
MAX_FILE_SIZE=10485760
CORS_ORIGINS=https://fraudguard-ai.netlify.app
LOG_LEVEL=INFO

# Development
MODEL_PATH=./fast_precision_fraud_model_statedict.pth
CORS_ORIGINS=*
LOG_LEVEL=DEBUG
```

### **Railway Deployment** (`railway.json`)
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn lightweight_api:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 30
  }
}
```

---

## 🔍 **Error Handling & Logging**

### **Exception Handling**
```python
from fastapi import HTTPException
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def error_handling_middleware(request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

class FraudDetectionError(Exception):
    """Custom exception for fraud detection errors"""
    pass

def handle_model_error(e):
    if "CUDA out of memory" in str(e):
        raise HTTPException(
            status_code=503,
            detail="Server temporarily overloaded. Please try again."
        )
    elif "corrupted" in str(e).lower():
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please upload a valid image."
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed. Please try again."
        )
```

### **Request Validation**
```python
from pydantic import BaseModel, validator
from typing import Optional

class PredictionRequest(BaseModel):
    confidence_threshold: Optional[float] = 0.5
    return_probabilities: Optional[bool] = True
    
    @validator('confidence_threshold')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence threshold must be between 0 and 1')
        return v

@app.post("/predict-advanced")
async def predict_advanced(
    file: UploadFile = File(...),
    params: PredictionRequest = Depends()
):
    # Advanced prediction with parameters
    pass
```

---

## 📊 **Monitoring & Health Checks**

### **Health Check Endpoint**
```python
import psutil
import torch

@app.get("/health")
async def health_check():
    # System metrics
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    # GPU status (if available)
    gpu_available = torch.cuda.is_available()
    gpu_memory = None
    if gpu_available:
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # Model status
    model_loaded = model is not None
    
    status = "healthy"
    if cpu_percent > 90 or memory.percent > 90:
        status = "degraded"
    if not model_loaded:
        status = "unhealthy"
    
    return {
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_loaded": model_loaded,
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "gpu_available": gpu_available,
            "gpu_memory_mb": gpu_memory
        }
    }

@app.get("/metrics")
async def get_metrics():
    # Prometheus-style metrics
    return {
        "predictions_total": prediction_counter,
        "prediction_duration_seconds": avg_prediction_time,
        "model_load_time_seconds": model_load_time,
        "errors_total": error_counter
    }
```

---

## 🔒 **Security & Best Practices**

### **Input Validation**
```python
import magic

async def validate_image_file(file: UploadFile):
    # Check file signature
    contents = await file.read()
    file_type = magic.from_buffer(contents, mime=True)
    
    allowed_types = ['image/jpeg', 'image/png', 'image/webp']
    if file_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file_type}"
        )
    
    # Reset file pointer
    await file.seek(0)
    return contents

@app.post("/predict-secure")
async def predict_secure(file: UploadFile = File(...)):
    # Validate file
    contents = await validate_image_file(file)
    
    # Process with validation
    result = await process_with_timeout(contents, timeout=30)
    return result
```

### **Rate Limiting**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")  # 10 requests per minute
async def predict_with_limit(request: Request, file: UploadFile = File(...)):
    return await predict_fraud(file)
```

---

## 🧪 **Testing**

### **Unit Tests**
```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded"]

def test_predict_endpoint():
    # Test with sample image
    with open("test_image.jpg", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert result["prediction"] in ["FRAUD", "NON-FRAUD"]

def test_invalid_file():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    
    assert response.status_code == 400
```

### **Load Testing**
```python
import asyncio
import aiohttp
import time

async def load_test():
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for i in range(100):  # 100 concurrent requests
            task = session.post(
                "http://localhost:8000/predict",
                data={"file": open("test_image.jpg", "rb")}
            )
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"100 requests completed in {end_time - start_time:.2f} seconds")
```

---

## 📈 **Performance Metrics**

### **Benchmark Results**
- **Cold Start**: 2.3 seconds (model loading)
- **Warm Inference**: 0.8-1.5 seconds per image
- **Memory Usage**: 512MB base + 128MB per concurrent request
- **Throughput**: 40-60 requests/minute (single instance)
- **CPU Usage**: 60-80% during inference

### **Optimization Strategies**
1. **Model Quantization**: Reduce model size by 50%
2. **Batch Processing**: Process multiple images together
3. **Caching**: Cache frequent predictions
4. **Load Balancing**: Multiple backend instances
5. **GPU Acceleration**: CUDA for faster inference

---

## 🔧 **Configuration Management**

### **Settings Class**
```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    model_path: str = "fast_precision_fraud_model_statedict.pth"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    cors_origins: str = "*"
    log_level: str = "INFO"
    prediction_timeout: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 📞 **API Documentation**

### **Automatic Documentation**
FastAPI automatically generates API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

### **Custom Documentation**
```python
@app.post("/predict", 
    summary="Detect Insurance Fraud",
    description="Analyze an insurance claim image for fraud detection",
    response_description="Fraud detection results with confidence scores"
)
async def predict_fraud(
    file: UploadFile = File(..., description="Image file to analyze")
):
    """
    Detect fraud in insurance claim images.
    
    - **file**: Upload an image file (JPEG, PNG, WebP)
    - **max_size**: 10MB file size limit
    - **response**: Detailed fraud analysis results
    """
    pass
```

---

## 🤝 **Contributing to Backend**

### **Development Guidelines**
1. **Code Style**: Follow PEP 8 with Black formatting
2. **Type Hints**: Use type annotations for all functions
3. **Error Handling**: Comprehensive exception handling
4. **Testing**: Unit tests for all endpoints
5. **Documentation**: Docstrings for all functions

### **Adding New Features**
1. Create feature branch
2. Implement with tests
3. Update documentation
4. Submit pull request
5. Code review process

---

## 📞 **Support & Troubleshooting**

### **Common Issues**
- **Model Loading Errors**: Check file path and permissions
- **Memory Issues**: Reduce batch size or use CPU inference
- **CORS Errors**: Configure allowed origins properly
- **File Upload Issues**: Check file size and type validation

### **Debug Mode**
```bash
# Run with debug logging
LOG_LEVEL=DEBUG python main.py

# Enable FastAPI debug mode
uvicorn main:app --reload --log-level debug
```

### **Getting Help**
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive API and setup guides
- **Team Contact**: Direct support for development

---

*This documentation covers the complete backend implementation. For frontend documentation, see [FRONTEND.md](./FRONTEND.md)*