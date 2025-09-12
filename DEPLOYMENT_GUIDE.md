# 🚀 FraudGuard AI - Deployment Guide

## � **Project Structure**

The application is now organized into separate frontend and backend folders for optimal deployment:

```
FraudGuard AI/
├── fraudguard-ai-frontend/     # React frontend application
│   ├── src/                    # Source code
│   ├── dist/                   # Built files (after npm run build)
│   ├── package.json           # Frontend dependencies
│   └── README.md              # Frontend documentation
├── fraudguard-ai-backend/      # Python ML API backend
│   ├── real_api.py            # FastAPI application
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile            # Container configuration
│   ├── railway.json          # Railway deployment config
│   ├── fast_precision_fraud_model_statedict.pth  # ML model
│   └── README.md             # Backend documentation
└── DEPLOYMENT_GUIDE.md        # This deployment guide
```

## �📋 **Complete Deployment Checklist**

Your FraudGuard AI application is ready for production! Here's everything you need to deploy it for free.

---

## 🎯 **Deployment Strategy**

- **Frontend**: Netlify (Free tier - perfect for React apps)
- **ML API**: Railway or Render (Free tier with GPU support)
- **Database**: Supabase (Free tier - if needed later)
- **Domain**: Free subdomain included

---

## 🔥 **Step 1: Frontend Deployment (Netlify)**

### 1.1 Prepare for Deployment

```bash
# Navigate to your project
cd "C:\Users\Kapil\Desktop\CTS\job-portal"

# Build the production version
npm run build
```

### 1.2 Deploy to Netlify

**Option A: Drag & Drop (Easiest)**
1. Go to [netlify.com](https://netlify.com)
2. Sign up/login with GitHub
3. Drag the `dist` folder to Netlify deploy area
4. Get your live URL (e.g., `https://amazing-app-123.netlify.app`)

**Option B: Git Integration (Recommended)**
1. Push your code to GitHub
2. Connect Netlify to your GitHub repo
3. Auto-deploy on every commit

### 1.3 Configure Netlify Settings

**Build Settings:**
- Build command: `npm run build`
- Publish directory: `dist`
- Node version: `18`

**Environment Variables:**
- `ML_API_URL`: Your ML API URL (from Step 2)

---

## 🤖 **Step 2: ML API Deployment**

### 2.1 Choose Platform

**Railway (Recommended - Better GPU support)**
- Go to [railway.app](https://railway.app)
- Sign up with GitHub
- Better for ML workloads

**Render (Alternative)**
- Go to [render.com](https://render.com)
- Sign up with GitHub
- Good free tier

### 2.2 Prepare ML API Files

Create these files in your `ml-api` folder:

**requirements.txt:**
```txt
fastapi==0.116.1
uvicorn==0.35.0
torch==2.7.1
torchvision==0.22.1
timm==1.0.19
pillow==11.3.0
python-multipart==0.0.20
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "real_api.py"]
```

**railway.json** (for Railway):
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "dockerfile"
  },
  "deploy": {
    "startCommand": "python real_api.py",
    "healthcheckPath": "/health"
  }
}
```

### 2.3 Deploy ML API

1. Create new service on Railway/Render
2. Connect your GitHub repo
3. Set environment variables:
   - `PORT`: 8000
   - `PYTHON_VERSION`: 3.11
4. Deploy and get your ML API URL

---

## ⚙️ **Step 3: Production Configuration**

### 3.1 Update Frontend API URL

In your deployed Netlify app, set environment variable:
```
ML_API_URL=https://your-ml-api.railway.app
```

### 3.2 Update Netlify Function

Create `netlify/functions/predict-fraud.js`:
```javascript
const ML_API_URL = process.env.ML_API_URL || 'https://your-ml-api.railway.app';

exports.handler = async (event, context) => {
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
      }
    };
  }

  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers: { 'Access-Control-Allow-Origin': '*' },
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  try {
    const { imageData, claimDetails } = JSON.parse(event.body);
    
    const response = await fetch(`${ML_API_URL}/predict-base64`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageData, claimDetails })
    });

    if (!response.ok) {
      throw new Error(`ML API error: ${response.status}`);
    }

    const result = await response.json();
    
    return {
      statusCode: 200,
      headers: { 'Access-Control-Allow-Origin': '*' },
      body: JSON.stringify(result)
    };

  } catch (error) {
    return {
      statusCode: 500,
      headers: { 'Access-Control-Allow-Origin': '*' },
      body: JSON.stringify({ 
        success: false, 
        error: 'Service unavailable',
        message: error.message 
      })
    };
  }
};
```

---

## 🧪 **Step 4: Testing & Monitoring**

### 4.1 Test Endpoints

**Health Check:**
```bash
curl https://your-ml-api.railway.app/health
```

**Frontend Test:**
1. Visit your Netlify URL
2. Upload test images
3. Verify predictions work

### 4.2 Monitor Performance

**Netlify:**
- Check build logs
- Monitor function usage
- Set up domain (optional)

**Railway/Render:**
- Monitor resource usage
- Check API response times
- Set up alerts

---

## 🎉 **Step 5: Going Live**

### 5.1 Final Checklist

- [ ] Frontend builds successfully
- [ ] ML API responds to health checks
- [ ] Image upload works end-to-end
- [ ] PDF generation works
- [ ] Error handling works
- [ ] Mobile responsive design
- [ ] Performance is acceptable

### 5.2 Custom Domain (Optional)

**Free Options:**
- Use Netlify subdomain: `your-app.netlify.app`
- Get free domain from Freenom
- Use GitHub Pages domain

**Paid Options:**
- Buy domain from Namecheap ($8-12/year)
- Connect to Netlify

---

## 💰 **Cost Breakdown (All FREE)**

| Service | Free Tier | Limits |
|---------|-----------|--------|
| **Netlify** | 100GB bandwidth | 300 build minutes |
| **Railway** | $5 credit monthly | Usually enough for hobby |
| **Render** | 750 hours/month | Good for small apps |
| **Supabase** | 500MB database | 2 projects |

**Total Monthly Cost: $0** ✨

---

## 🚨 **Common Issues & Solutions**

### Build Errors
- Check Node.js version (use 18)
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules`, run `npm install`

### ML API Timeout
- Optimize model loading
- Add health check endpoints
- Use smaller model size

### CORS Issues
- Set proper headers in Netlify functions
- Configure ML API CORS settings

---

## 📞 **Support Resources**

- **Netlify Docs**: [docs.netlify.com](https://docs.netlify.com)
- **Railway Docs**: [docs.railway.app](https://docs.railway.app)
- **Render Docs**: [render.com/docs](https://render.com/docs)

---

## 🎯 **Next Steps After Deployment**

1. **Analytics**: Add Google Analytics
2. **SEO**: Optimize meta tags
3. **Monitoring**: Set up error tracking
4. **Security**: Add rate limiting
5. **Features**: User authentication, history
6. **Scale**: Upgrade to paid tiers when needed

---

**Your FraudGuard AI is ready for the world! 🌍✨**