import { useEffect } from "react";
import { SignIn, useUser } from "@clerk/clerk-react";
import { useNavigate } from "react-router-dom";
import { Shield, Brain, CheckCircle, Building2, TrendingUp, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";

const AuthPage = () => {
  const navigate = useNavigate();
  
  // Development mode check
  const isDevelopment = !import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
  
  // Only use Clerk hooks if not in development mode
  const { user, isLoaded } = isDevelopment ? { user: null, isLoaded: true } : useUser();

  // Insurance and tech companies using fraud detection
  const partnerCompanies = [
    { name: "Allstate", logo: "🛡️" },
    { name: "State Farm", logo: "🏢" },
    { name: "Progressive", logo: "📈" },
    { name: "GEICO", logo: "🦎" },
    { name: "Liberty Mutual", logo: "🗽" },
    { name: "Farmers", logo: "🚜" },
    { name: "USAA", logo: "⭐" },
    { name: "Nationwide", logo: "🌐" },
    { name: "Travelers", logo: "🧳" },
    { name: "Hartford", logo: "🦌" },
    { name: "Chubb", logo: "🔒" },
    { name: "AXA", logo: "🔷" }
  ];

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

          {/* Animated Scrolling Banner for Dev Mode */}
          <div className="p-6 bg-white/10 backdrop-blur-sm rounded-lg border border-white/20 mb-8">
            <h3 className="text-lg font-semibold mb-4">Trusted by Leading Insurance Companies</h3>
            <div className="relative overflow-hidden">
              <div className="flex animate-scroll whitespace-nowrap">
                {[...partnerCompanies, ...partnerCompanies].map((company, index) => (
                  <div key={index} className="flex items-center mx-8 flex-shrink-0">
                    <span className="text-2xl mr-2">{company.logo}</span>
                    <span className="text-gray-300 font-medium">{company.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
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
            <h3 className="text-lg font-semibold mb-4 text-center">Trusted by Leading Insurance Companies</h3>
            
            {/* Animated Scrolling Banner */}
            <div className="relative overflow-hidden">
              <div className="flex animate-scroll whitespace-nowrap">
                {[...partnerCompanies, ...partnerCompanies].map((company, index) => (
                  <div key={index} className="flex items-center mx-8 flex-shrink-0">
                    <span className="text-2xl mr-2">{company.logo}</span>
                    <span className="text-gray-300 font-medium">{company.name}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="mt-4 space-y-3">
              <div className="flex items-center gap-3">
                <TrendingUp className="text-green-400" size={20} />
                <span className="text-sm text-gray-300">Real-time fraud probability analysis</span>
              </div>
              <div className="flex items-center gap-3">
                <Building2 className="text-blue-400" size={20} />
                <span className="text-sm text-gray-300">Enterprise-grade security & compliance</span>
              </div>
              <div className="flex items-center gap-3">
                <Zap className="text-yellow-400" size={20} />
                <span className="text-sm text-gray-300">Instant PDF reports & analytics dashboard</span>
              </div>
            </div>
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