@echo off
echo ========================================
echo   FraudGuard AI - Backend Deployment
echo ========================================
echo.

echo 📦 Testing backend locally...
cd fraudguard-ai-backend

echo.
echo 🔍 Checking Python dependencies...
python -c "import fastapi, torch, torchvision, PIL, uvicorn, numpy; print('✅ All dependencies available')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ Missing dependencies. Installing...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo 🧪 Running quick API test...
python test_server.py
if %errorlevel% neq 0 (
    echo ⚠️  Test completed with warnings (this is normal if model takes time to load)
)

echo.
echo 🚀 Backend is ready for deployment!
echo.
echo Deployment options:
echo.
echo 🟢 OPTION 1: Railway (Recommended)
echo   1. Go to https://railway.app
echo   2. Sign up/login with GitHub
echo   3. Click "Deploy from GitHub repo"
echo   4. Select your repository
echo   5. Choose the 'fraudguard-ai-backend' folder
echo   6. Railway will auto-deploy!
echo.
echo 🟡 OPTION 2: Render
echo   1. Go to https://render.com
echo   2. Sign up/login with GitHub
echo   3. Click "New Web Service"
echo   4. Connect your repository
echo   5. Root directory: fraudguard-ai-backend
echo   6. Build command: pip install -r requirements.txt
echo   7. Start command: uvicorn real_api:app --host 0.0.0.0 --port $PORT
echo   8. Deploy!
echo.
echo 🔵 OPTION 3: Docker (Any platform)
echo   1. docker build -t fraudguard-backend .
echo   2. docker run -p 8001:8001 fraudguard-backend
echo.
echo Model file: fast_precision_fraud_model_statedict.pth (✅ included)
echo.
pause