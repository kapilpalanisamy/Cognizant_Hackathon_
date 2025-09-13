# ⚛️ Frontend Documentation - FraudGuard AI

## 🎯 **Overview**

The FraudGuard AI frontend is a modern React application built with **Vite**, **Tailwind CSS**, and **Clerk Authentication**. It provides an intuitive interface for insurance professionals to upload claim images, analyze fraud risk, and generate comprehensive reports.

---

## 🏗️ **Architecture & Tech Stack**

### **Core Technologies**
- **React 18**: Latest React with concurrent features
- **Vite**: Ultra-fast build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **Clerk**: Modern authentication and user management
- **Supabase**: Backend-as-a-Service for data storage

### **Key Libraries**
```json
{
  "react": "^18.2.0",
  "vite": "^4.4.5",
  "tailwindcss": "^3.3.0",
  "@clerk/clerk-react": "^4.23.2",
  "jspdf": "^2.5.1",
  "lucide-react": "^0.263.1",
  "react-router-dom": "^6.15.0"
}
```

---

## 📁 **Project Structure**

```
fraudguard-ai-frontend/
├── 📁 public/                    # Static assets
│   ├── logo.png                 # Application logo
│   ├── logo-dark.png           # Dark theme logo
│   └── _redirects              # Netlify redirects
├── 📁 src/
│   ├── 📁 components/           # Reusable UI components
│   │   ├── header.jsx          # Navigation header
│   │   ├── protected-route.jsx  # Auth route protection
│   │   ├── theme-provider.jsx   # Theme management
│   │   └── 📁 ui/              # Shadcn/ui components
│   ├── 📁 hooks/               # Custom React hooks
│   │   └── use-fetch.js        # API fetching hook
│   ├── 📁 layouts/             # Page layouts
│   │   └── app-layout.jsx      # Main app layout
│   ├── 📁 lib/                 # Utility libraries
│   │   └── utils.js            # Helper functions
│   ├── 📁 pages/               # Application pages
│   │   ├── auth.jsx            # Authentication page
│   │   ├── fraud-detection.jsx # Main analysis page
│   │   └── landing.jsx         # Landing page
│   ├── 📁 utils/               # Utility functions
│   │   ├── pdfGenerator.js     # PDF report generation
│   │   └── supabase.js         # Supabase client
│   ├── App.jsx                 # Root component
│   ├── main.jsx               # Application entry point
│   └── index.css              # Global styles
├── 📁 netlify/
│   └── 📁 functions/           # Serverless functions
│       ├── predict-fraud.js    # ML API integration
│       └── wake-ml-service.js  # Cold start handling
├── package.json               # Dependencies & scripts
├── vite.config.js            # Vite configuration
├── tailwind.config.js        # Tailwind configuration
└── netlify.toml              # Netlify deployment config
```

---

## 🚀 **Getting Started**

### **Prerequisites**
- Node.js 18+ and npm/yarn
- Modern web browser
- Internet connection for API calls

### **Installation**
```bash
# Clone the repository
git clone https://github.com/kapilpalanisamy/Cognizant_Hackathon_.git

# Navigate to frontend
cd Cognizant_Hackathon_/fraudguard-ai-frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### **Environment Setup**
Create a `.env` file in the root directory:
```env
# Supabase Configuration
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key

# Clerk Authentication
VITE_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key

# API Configuration
VITE_ML_API_URL=https://fraudguard-ai-backend.onrender.com
```

---

## 🎨 **UI Components & Design System**

### **Design Principles**
- **Accessibility First**: WCAG 2.1 compliant
- **Mobile Responsive**: Works on all device sizes
- **Dark/Light Theme**: Automatic theme switching
- **Professional Aesthetics**: Insurance industry standards

### **Component Library**
Built using **Shadcn/ui** components with custom modifications:

#### **Core Components**
```jsx
// Button variations
<Button variant="default">Primary Action</Button>
<Button variant="outline">Secondary Action</Button>
<Button variant="ghost">Subtle Action</Button>

// Input components
<Input type="file" accept="image/*" />
<Textarea placeholder="Additional notes..." />

// Layout components
<Card>
  <CardHeader>
    <CardTitle>Section Title</CardTitle>
  </CardHeader>
  <CardContent>Content here</CardContent>
</Card>
```

#### **Custom Components**

**Header Component** (`components/header.jsx`)
```jsx
export function Header() {
  return (
    <header className="border-b">
      <div className="container mx-auto px-4 py-3">
        <nav className="flex items-center justify-between">
          <Logo />
          <UserButton />
        </nav>
      </div>
    </header>
  );
}
```

**Protected Route** (`components/protected-route.jsx`)
```jsx
export function ProtectedRoute({ children }) {
  const { isLoaded, isSignedIn } = useAuth();
  
  if (!isLoaded) return <LoadingSpinner />;
  if (!isSignedIn) return <RedirectToSignIn />;
  
  return children;
}
```

---

## 📄 **Page Components**

### **Landing Page** (`pages/landing.jsx`)
- **Hero Section**: Value proposition and CTA
- **Features**: Key capabilities showcase
- **Benefits**: Business impact highlights
- **Call-to-Action**: Sign up/Demo request

### **Authentication Page** (`pages/auth.jsx`)
- **Sign In/Up Forms**: Clerk-powered authentication
- **Social Login**: Google, Microsoft integration
- **Password Reset**: Self-service password recovery
- **Animated Features**: Engaging user onboarding

### **Fraud Detection Page** (`pages/fraud-detection.jsx`)
**Core Features:**
- **Drag & Drop Upload**: Intuitive file selection
- **Real-time Analysis**: Live progress indicators
- **Results Display**: Visual fraud/non-fraud indicators
- **Confidence Scoring**: Percentage-based reliability
- **PDF Generation**: Professional report creation

**Component Structure:**
```jsx
export function FraudDetectionPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleFileUpload = async (file) => {
    setIsAnalyzing(true);
    try {
      const result = await analyzeFraud(file);
      setPrediction(result);
    } catch (error) {
      handleError(error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="container mx-auto p-6">
      <FileUploadZone onFileSelect={handleFileUpload} />
      {isAnalyzing && <AnalysisProgress />}
      {prediction && <ResultsDisplay prediction={prediction} />}
    </div>
  );
}
```

---

## 🔌 **API Integration**

### **Serverless Functions**
Located in `netlify/functions/`:

#### **Fraud Prediction** (`predict-fraud.js`)
```javascript
exports.handler = async (event, context) => {
  try {
    // Parse multipart form data
    const formData = parseFormData(event.body);
    
    // Call ML API
    const response = await fetch('ML_API_URL/predict', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    
    return {
      statusCode: 200,
      headers: corsHeaders,
      body: JSON.stringify(result)
    };
  } catch (error) {
    return handleError(error);
  }
};
```

#### **Cold Start Handling** (`wake-ml-service.js`)
```javascript
exports.handler = async (event, context) => {
  try {
    // Wake up sleeping Render service
    await fetch('ML_API_URL/health');
    
    return {
      statusCode: 200,
      body: JSON.stringify({ message: 'Service awake' })
    };
  } catch (error) {
    return { statusCode: 503, body: 'Service unavailable' };
  }
};
```

### **API Error Handling**
```jsx
const handleApiError = (error) => {
  if (error.status === 503) {
    // Handle cold start
    showMessage('Waking up ML service, please wait...');
    wakeService().then(() => retryAnalysis());
  } else {
    showError('Analysis failed. Please try again.');
  }
};
```

---

## 📊 **State Management**

### **React Hooks Strategy**
- **useState**: Local component state
- **useEffect**: Side effects and API calls
- **useContext**: Global theme and auth state
- **Custom Hooks**: Reusable logic patterns

#### **Custom Hooks**

**useFetch Hook** (`hooks/use-fetch.js`)
```jsx
export function useFetch(url, options = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(url, options);
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [url, options]);

  return { data, loading, error, refetch: fetchData };
}
```

---

## 🎯 **Features Implementation**

### **File Upload & Validation**
```jsx
const validateFile = (file) => {
  const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
  const maxSize = 10 * 1024 * 1024; // 10MB
  
  if (!validTypes.includes(file.type)) {
    throw new Error('Please upload a valid image file (JPEG, PNG, WebP)');
  }
  
  if (file.size > maxSize) {
    throw new Error('File size must be less than 10MB');
  }
  
  return true;
};
```

### **Real-time Progress Indicators**
```jsx
const AnalysisProgress = () => {
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('Initializing...');

  useEffect(() => {
    const timer = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return prev;
        return prev + Math.random() * 15;
      });
    }, 500);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="text-center space-y-4">
      <Progress value={progress} className="w-full" />
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );
};
```

### **PDF Report Generation**
Using **jsPDF** for client-side PDF creation:
```jsx
import { generateFraudReport } from '../utils/pdfGenerator';

const downloadReport = async () => {
  try {
    const filename = await generateFraudReport(prediction, selectedFile);
    showSuccess(`Report downloaded: ${filename}`);
  } catch (error) {
    showError('Failed to generate report');
  }
};
```

---

## 🎨 **Styling & Theming**

### **Tailwind Configuration**
```javascript
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          900: '#1e3a8a',
        },
        fraud: {
          light: '#fef2f2',
          DEFAULT: '#dc2626',
          dark: '#991b1b',
        },
        safe: {
          light: '#f0fdf4',
          DEFAULT: '#16a34a',
          dark: '#15803d',
        }
      }
    }
  },
  plugins: [require("@tailwindcss/forms")]
};
```

### **CSS Custom Properties**
```css
:root {
  --primary: 222.2 84% 4.9%;
  --primary-foreground: 210 40% 98%;
  --fraud-indicator: 0 84% 60%;
  --safe-indicator: 142 76% 36%;
}

.dark {
  --primary: 210 40% 98%;
  --primary-foreground: 222.2 84% 4.9%;
}
```

---

## 🔒 **Authentication & Security**

### **Clerk Integration**
```jsx
import { ClerkProvider, useAuth } from '@clerk/clerk-react';

// App wrapper
function App() {
  return (
    <ClerkProvider publishableKey={process.env.VITE_CLERK_PUBLISHABLE_KEY}>
      <Router>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/auth/*" element={<AuthPage />} />
          <Route path="/analysis" element={
            <ProtectedRoute>
              <FraudDetectionPage />
            </ProtectedRoute>
          } />
        </Routes>
      </Router>
    </ClerkProvider>
  );
}
```

### **Security Features**
- **Route Protection**: Authenticated routes only
- **File Validation**: Client and server-side validation
- **CORS Headers**: Proper cross-origin handling
- **CSP Headers**: Content Security Policy implementation
- **Rate Limiting**: API call throttling

---

## 📱 **Responsive Design**

### **Breakpoint Strategy**
```css
/* Mobile First Approach */
.container {
  @apply px-4;                    /* Mobile: 16px padding */
}

@screen sm {                      /* 640px+ */
  .container { @apply px-6; }
}

@screen lg {                      /* 1024px+ */
  .container { @apply px-8; }
}

@screen xl {                      /* 1280px+ */
  .container { @apply px-12; }
}
```

### **Mobile Optimizations**
- **Touch-friendly buttons**: 44px minimum touch targets
- **Optimized images**: WebP format with fallbacks
- **Gesture support**: Swipe navigation where appropriate
- **Performance**: Lazy loading and code splitting

---

## ⚡ **Performance Optimization**

### **Build Optimization**
```javascript
// vite.config.js
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          ui: ['@clerk/clerk-react', 'lucide-react'],
          pdf: ['jspdf']
        }
      }
    }
  },
  plugins: [react()]
});
```

### **Performance Features**
- **Code Splitting**: Automatic route-based splitting
- **Lazy Loading**: Dynamic imports for non-critical components
- **Image Optimization**: Responsive images with proper sizing
- **Bundle Analysis**: Webpack bundle analyzer integration

---

## 🧪 **Development & Testing**

### **Development Scripts**
```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext js,jsx --fix",
    "format": "prettier --write ."
  }
}
```

### **Code Quality Tools**
- **ESLint**: JavaScript/React linting
- **Prettier**: Code formatting
- **Husky**: Git hooks for quality checks
- **Commitlint**: Conventional commit messages

---

## 🚀 **Deployment**

### **Netlify Configuration**
```toml
# netlify.toml
[build]
  publish = "dist"
  command = "npm run build"

[build.environment]
  NODE_VERSION = "18"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-XSS-Protection = "1; mode=block"
```

### **Production Build**
```bash
# Build for production
npm run build

# Preview production build locally
npm run preview

# Deploy to Netlify
netlify deploy --prod --dir=dist
```

---

## 🔧 **Configuration Files**

### **Vite Configuration**
```javascript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    open: true
  }
});
```

### **JSConfig for Path Mapping**
```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"],
      "@/components/*": ["./src/components/*"],
      "@/utils/*": ["./src/utils/*"]
    }
  }
}
```

---

## 📊 **Monitoring & Analytics**

### **Performance Monitoring**
- **Web Vitals**: Core web vitals tracking
- **Error Tracking**: Client-side error reporting
- **User Analytics**: Usage patterns and flows
- **Performance Metrics**: Load times and interactions

---

## 🤝 **Contributing to Frontend**

### **Development Guidelines**
1. **Component Creation**: Use functional components with hooks
2. **Styling**: Tailwind utility classes preferred
3. **State Management**: Keep state close to where it's used
4. **Error Handling**: Comprehensive error boundaries
5. **Accessibility**: Follow WCAG guidelines

### **Pull Request Process**
1. Create feature branch from `main`
2. Implement changes with tests
3. Run linting and formatting
4. Submit PR with description
5. Review and merge process

---

## 📞 **Support & Troubleshooting**

### **Common Issues**
- **Build Failures**: Check Node.js version (18+)
- **Authentication Issues**: Verify Clerk configuration
- **API Errors**: Check network connectivity and service status
- **Styling Issues**: Clear browser cache and rebuild

### **Getting Help**
- **GitHub Issues**: Technical problems and bugs
- **Documentation**: Comprehensive guides and examples
- **Team Contact**: Direct support for development questions

---

*This documentation covers the complete frontend implementation. For backend documentation, see [BACKEND.md](./BACKEND.md)*