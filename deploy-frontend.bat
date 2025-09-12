@echo off
echo ========================================
echo   FraudGuard AI - Frontend Deployment
echo ========================================
echo.

echo 📦 Installing dependencies...
cd fraudguard-ai-frontend
call npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ Dependencies installed successfully

echo.
echo 🔨 Building production version...
call npm run build
if %errorlevel% neq 0 (
    echo ❌ Build failed
    pause
    exit /b 1
)
echo ✅ Build completed successfully

echo.
echo 🚀 Frontend is ready for deployment!
echo.
echo Next steps:
echo 1. Go to https://netlify.com
echo 2. Drag and drop the 'dist' folder to deploy
echo 3. Or connect your Git repository for automatic deployments
echo.
echo Build output location: fraudguard-ai-frontend/dist/
echo.
pause