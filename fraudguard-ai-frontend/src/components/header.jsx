import { Link } from "react-router-dom";
import {
  SignedIn,
  UserButton,
  useUser,
} from "@clerk/clerk-react";
import { Button } from "./ui/button";
import { Shield, Brain } from "lucide-react";

const Header = () => {
  // Development mode check
  const isDevelopment = !import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
  
  // Only use Clerk hooks in production mode
  const { user } = isDevelopment ? { user: null } : useUser();

  return (
    <>
      <nav className="py-4 flex justify-between items-center">
  <Link to="/risk-assessment">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl">
              <Shield size={28} className="text-white" />
            </div>
            <div>
              <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                FraudGuard AI
              </span>
              <p className="text-xs text-slate-500 -mt-1">Advanced Fraud Detection</p>
            </div>
          </div>
        </Link>

        <div className="flex gap-4 items-center">
          {isDevelopment ? (
            // Development mode - simplified navigation
            <>
              <div className="flex items-center gap-2 px-3 py-1 bg-amber-100 text-amber-800 rounded-full text-sm font-medium">
                <span>🛠️ Development Mode</span>
              </div>
            </>
          ) : (
            // Production mode with Clerk authentication
            <>
              <SignedIn>
                <div className="flex items-center gap-2 px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                  <span>✅ Authenticated</span>
                </div>
                <UserButton
                  appearance={{
                    elements: {
                      avatarBox: "w-10 h-10",
                    },
                  }}
                  afterSignOutUrl="/"
                >
                  <UserButton.MenuItems>
                    <UserButton.Link
                      label="AI Fraud Detection"
                      labelIcon={<Brain size={15} />}
                      href="/risk-assessment"
                    />
                    <UserButton.Action label="manageAccount" />
                  </UserButton.MenuItems>
                </UserButton>
              </SignedIn>
            </>
          )}
        </div>
      </nav>
    </>
  );
};

export default Header;
