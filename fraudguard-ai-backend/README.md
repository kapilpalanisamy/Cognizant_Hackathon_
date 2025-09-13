# FraudGuard AI Backend

FastAPI-based ML backend for insurance fraud detection using PyTorch and EfficientNet.

## Features

- **PyTorch Model**: EfficientNet-B1 based FinalModelDetector
- **High Performance**: 87.9% precision, 91.4% accuracy
- **Fast Inference**: Optimized for real-time predictions
- **CORS Enabled**: Ready for web application integration
- **Docker Support**: Container-ready deployment
- **Cloud Ready**: Configured for Railway, Render, and other platforms

## Model Details

- **Architecture**: EfficientNet-B1 backbone
- **Training Data**: Insurance claim images (fraud/non-fraud)
- **Performance Metrics**:
  - Precision: 87.9%
  - Accuracy: 91.4%
  - Optimized for fraud detection precision
- **Input**: 224x224 RGB images
- **Output**: Fraud probability score (0-1)

## Tech Stack

- **Framework**: FastAPI
- **ML Library**: PyTorch
- **Image Processing**: PIL, torchvision
- **Model**: EfficientNet-B1 (timm)
- **Deployment**: Uvicorn ASGI server
- **Containerization**: Docker

## Quick Start

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start development server**:
   ```bash
   uvicorn real_api:app --reload --port 8001
   ```

3. **Test the API**:
   ```bash
   python test_server.py
   ```

### Docker Deployment

1. **Build container**:
   ```bash
   docker build -t fraudguard-ai-backend .
   ```

2. **Run container**:
   ```bash
   docker run -p 8001:8001 fraudguard-ai-backend
   ```

## API Endpoints

### POST `/predict`

Predict fraud probability for an uploaded image.

**Request**:
```bash
curl -X POST "http://localhost:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{"image": "base64_encoded_image_data"}'
```

**Response**:
```json
{
  "prediction": "FRAUD" | "NON-FRAUD",
  "confidence": 0.95,
  "processing_time": 0.234
}
```

### GET `/health`

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Deployment

### Railway (Recommended)

1. Connect your repository to Railway
2. Railway auto-detects the Python app
3. Environment variables are automatically configured
4. Deploy with zero configuration!

### Render

1. Connect your repository to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn real_api:app --host 0.0.0.0 --port $PORT`
4. Deploy!

### Docker Deployment

The included `Dockerfile` provides a production-ready container:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "real_api:app", "--host", "0.0.0.0", "--port", "8001"]
```

## Model Files

- `final_model_statedict.pth`: PyTorch state dict (production)
- `final_model.pth`: Full model file (development)
- `convert_model.py`: Script to convert between formats

## Configuration

### Environment Variables

- `PORT`: Server port (default: 8001)
- `MODEL_PATH`: Path to model file (default: auto-detected)

### Model Loading

The API automatically detects and loads the appropriate model format:

1. State dict format (`.pth` with state dict)
2. Full model format (`.pth` with complete model)

## Performance

- **Cold Start**: ~2-3 seconds (model loading)
- **Inference Time**: ~200-300ms per image
- **Memory Usage**: ~500MB (model in memory)
- **Concurrent Requests**: Supported via FastAPI async

## Monitoring

### Health Checks

```bash
curl http://localhost:8001/health
```

### Logs

```bash
# View application logs
docker logs <container_id>

# Follow logs in real-time
docker logs -f <container_id>
```

## Troubleshooting

### Common Issues

1. **Model not loading**: Check file paths and format compatibility
2. **Memory errors**: Ensure sufficient RAM (2GB+ recommended)
3. **Slow inference**: Check CPU/GPU availability

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python test_server.py`
5. Submit a pull request

## License

MIT License - see LICENSE file for details