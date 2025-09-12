import { Link } from "react-router-dom";
import {
  SignedIn,
  UserButton,
  useUser,
} from "@clerk/clerk-react";
import { Button } from "./ui/button";
import { Shield, Upload, BarChart3 } from "lucide-react";

const Header = () => {
  // Development mode check
  const isDevelopment = !import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
  const { user } = isDevelopment ? { user: null } : useUser();

  return (
    <>
      <nav className="py-4 flex justify-between items-center">
        <Link to="/dashboard">
          <div className="flex items-center gap-2">
            <Shield size={32} className="text-blue-600" />
            <span className="text-2xl font-bold">FraudGuard AI</span>
          </div>
        </Link>

        <div className="flex gap-4">
          {isDevelopment ? (
            // Development mode - show navigation without auth
            <>
              <Link to="/dashboard">
                <Button variant="outline" className="rounded-full">
                  <BarChart3 size={20} className="mr-2" />
                  Dashboard
                </Button>
              </Link>
              <Link to="/fraud-detection">
                <Button variant="default" className="rounded-full">
                  <Upload size={20} className="mr-2" />
                  Analyze Image
                </Button>
              </Link>
            </>
          ) : (
            // Production mode with Clerk authentication
            <>
              <SignedIn>
                <Link to="/dashboard">
                  <Button variant="outline" className="rounded-full">
                    <BarChart3 size={20} className="mr-2" />
                    Dashboard
                  </Button>
                </Link>
                <Link to="/fraud-detection">
                  <Button variant="default" className="rounded-full">
                    <Upload size={20} className="mr-2" />
                    Analyze Image
                  </Button>
                </Link>
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
                      label="Dashboard"
                      labelIcon={<BarChart3 size={15} />}
                      href="/dashboard"
                    />
                    <UserButton.Link
                      label="Fraud Detection"
                      labelIcon={<Shield size={15} />}
                      href="/fraud-detection"
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
