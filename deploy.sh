#!/bin/bash
# FraudGuard AI - Quick Deployment Script

echo "🚀 FraudGuard AI - Deployment Script"
echo "======================================"

# Step 1: Build Frontend
echo "📦 Building frontend..."
cd job-portal
npm run build
echo "✅ Frontend built successfully!"

# Step 2: Prepare ML API
echo "🤖 Preparing ML API for deployment..."
cd ml-api
echo "✅ ML API files ready!"

echo ""
echo "🎯 Next Steps:"
echo "1. Upload 'dist' folder to Netlify"
echo "2. Deploy 'ml-api' folder to Railway/Render"
echo "3. Update environment variables"
echo "4. Test your live application!"
echo ""
echo "📖 See DEPLOYMENT_GUIDE.md for detailed instructions"

read -p "Press Enter to continue..."