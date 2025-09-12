@echo off
echo ==========================================
echo      🚀 FraudGuard AI - Full Deployment
echo ==========================================
echo.
echo This script will prepare both frontend and backend for deployment.
echo.

:menu
echo Please choose an option:
echo.
echo 1. Deploy Frontend only (React app to Netlify)
echo 2. Deploy Backend only (ML API to Railway/Render)
echo 3. Deploy Both (Full application)
echo 4. Quick test everything
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto frontend
if "%choice%"=="2" goto backend
if "%choice%"=="3" goto both
if "%choice%"=="4" goto test
if "%choice%"=="5" goto exit
echo Invalid choice. Please try again.
goto menu

:frontend
echo.
echo 🎨 Deploying Frontend...
call deploy-frontend.bat
goto menu

:backend
echo.
echo 🤖 Deploying Backend...
call deploy-backend.bat
goto menu

:both
echo.
echo 📱 Deploying Frontend...
call deploy-frontend.bat
echo.
echo 🤖 Deploying Backend...
call deploy-backend.bat
echo.
echo ✅ Both frontend and backend are ready!
echo.
echo 📋 Final steps:
echo 1. Deploy frontend 'dist' folder to Netlify
echo 2. Deploy backend to Railway or Render
echo 3. Update frontend API URL if needed
echo.
goto menu

:test
echo.
echo 🧪 Running quick tests...
echo.
echo Testing frontend build...
cd fraudguard-ai-frontend
call npm run build >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Frontend builds successfully
) else (
    echo ❌ Frontend build failed
)
cd ..

echo.
echo Testing backend...
cd fraudguard-ai-backend
python -c "import fastapi, torch; print('✅ Backend dependencies OK')" 2>nul
if %errorlevel% equ 0 (
    echo ✅ Backend dependencies available
) else (
    echo ❌ Backend missing dependencies
)
cd ..

echo.
echo 📊 Test completed!
goto menu

:exit
echo.
echo 👋 Thank you for using FraudGuard AI!
echo.
echo 📚 Documentation:
echo   - Frontend: fraudguard-ai-frontend/README.md
echo   - Backend: fraudguard-ai-backend/README.md
echo   - Deployment: DEPLOYMENT_GUIDE.md
echo.
echo 🌐 Helpful links:
echo   - Netlify: https://netlify.com
echo   - Railway: https://railway.app
echo   - Render: https://render.com
echo.
pause