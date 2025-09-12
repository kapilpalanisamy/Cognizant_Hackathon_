import { useEffect } from "react";
import { SignIn, useUser } from "@clerk/clerk-react";
import { useNavigate } from "react-router-dom";
import { Shield, Brain, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

const AuthPage = () => {
  const navigate = useNavigate();
  
  // Development mode check
  const isDevelopment = !import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
  
  // Only use Clerk hooks if not in development mode
  const { user, isLoaded } = isDevelopment ? { user: null, isLoaded: true } : useUser();

  // Redirect to fraud detection if already signed in
  useEffect(() => {
    if (isDevelopment) {
      // In development mode, go directly to fraud detection
      return;
    }
    
    if (isLoaded && user) {
  navigate("/risk-assessment");
    }
  }, [isLoaded, user, navigate, isDevelopment]);

  // Development mode - simple redirect to dashboard
  if (isDevelopment) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center p-4">
        <div className="max-w-4xl w-full text-center text-white space-y-8">
          <div className="flex items-center justify-center gap-3 mb-8">
            <Shield size={48} className="text-blue-400" />
            <h1 className="text-4xl font-bold">FraudGuard AI</h1>
          </div>
          
          <h2 className="text-3xl font-semibold mb-4">
            Development Mode
          </h2>
          <p className="text-xl text-gray-300 mb-8">
            Authentication is disabled for development. Click below to access the AI fraud detection system.
          </p>
          
          <Button 
            size="lg" 
            className="bg-blue-600 hover:bg-blue-700" 
            onClick={() => navigate("/risk-assessment")}
          >
            Start AI Analysis
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center p-4">
      <div className="max-w-6xl w-full grid lg:grid-cols-2 gap-12 items-center">
        
        {/* Left Side - Branding & Features */}
        <div className="text-white space-y-8">
          <div className="flex items-center gap-3">
            <Shield size={48} className="text-blue-400" />
            <h1 className="text-4xl font-bold">FraudGuard AI</h1>
          </div>
          
          <div className="space-y-4">
            <h2 className="text-3xl font-semibold">
              Advanced Insurance Fraud Detection
            </h2>
            <p className="text-xl text-gray-300">
              Powered by cutting-edge AI technology to identify fraudulent claims with 91.4% accuracy
            </p>
          </div>

          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <Brain className="text-green-400" size={24} />
              <span className="text-lg">Real-time AI Analysis</span>
            </div>
            <div className="flex items-center gap-3">
              <CheckCircle className="text-green-400" size={24} />
              <span className="text-lg">91.4% Accuracy Rate</span>
            </div>
            <div className="flex items-center gap-3">
              <Shield className="text-green-400" size={24} />
              <span className="text-lg">Secure & Professional</span>
            </div>
          </div>

          <div className="p-6 bg-white/10 backdrop-blur-sm rounded-lg border border-white/20">
            <h3 className="text-lg font-semibold mb-2">What you'll get access to:</h3>
            <ul className="space-y-2 text-gray-300">
              <li>• Upload and analyze vehicle damage images</li>
              <li>• Get instant fraud probability scores</li>
              <li>• Generate detailed PDF reports</li>
              <li>• Access to analytics dashboard</li>
            </ul>
          </div>
        </div>

        {/* Right Side - Authentication */}
        <div className="flex justify-center lg:justify-end">
          <div className="w-full max-w-md">
            <SignIn
              signUpForceRedirectUrl="/dashboard"
              fallbackRedirectUrl="/dashboard"
              appearance={{
                elements: {
                  formButtonPrimary: "bg-blue-600 hover:bg-blue-700",
                  card: "shadow-2xl",
                  headerTitle: "text-2xl font-bold",
                  headerSubtitle: "text-gray-600"
                }
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuthPage;