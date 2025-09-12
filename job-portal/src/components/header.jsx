import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  SignedIn,
  SignedOut,
  UserButton,
  SignIn,
  useUser,
} from "@clerk/clerk-react";
import { Button } from "./ui/button";
import { Shield, Upload, BarChart3 } from "lucide-react";

const Header = () => {
  const [showSignIn, setShowSignIn] = useState(false);
  const [search, setSearch] = useSearchParams();
  
  // Development mode check
  const isDevelopment = !import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
  const { user } = isDevelopment ? { user: null } : useUser();

  useEffect(() => {
    if (search.get("sign-in")) {
      setShowSignIn(true);
    }
  }, [search]);

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      setShowSignIn(false);
      setSearch({});
    }
  };

  return (
    <>
      <nav className="py-4 flex justify-between items-center">
        <Link to="/">
          <div className="flex items-center gap-2">
            <Shield size={32} className="text-blue-600" />
            <span className="text-2xl font-bold">FraudGuard AI</span>
          </div>
        </Link>

        <div className="flex gap-4">
          {isDevelopment ? (
            // Development mode - show navigation without auth
            <>
              <Link to="/fraud-detection">
                <Button variant="default" className="rounded-full">
                  <Upload size={20} className="mr-2" />
                  Analyze Image
                </Button>
              </Link>
              <Link to="/dashboard">
                <Button variant="outline" className="rounded-full">
                  <BarChart3 size={20} className="mr-2" />
                  Dashboard
                </Button>
              </Link>
            </>
          ) : (
            // Production mode with Clerk authentication
            <>
              <SignedOut>
                <Button variant="outline" onClick={() => setShowSignIn(true)}>
                  Login
                </Button>
              </SignedOut>
              <SignedIn>
                <Link to="/fraud-detection">
                  <Button variant="default" className="rounded-full">
                    <Upload size={20} className="mr-2" />
                    Analyze Image
                  </Button>
                </Link>
                <Link to="/dashboard">
                  <Button variant="outline" className="rounded-full">
                    <BarChart3 size={20} className="mr-2" />
                    Dashboard
                  </Button>
                </Link>
                <UserButton
                  appearance={{
                    elements: {
                      avatarBox: "w-10 h-10",
                    },
                  }}
                >
                  <UserButton.MenuItems>
                    <UserButton.Link
                      label="Fraud Detection"
                      labelIcon={<Shield size={15} />}
                      href="/fraud-detection"
                    />
                    <UserButton.Link
                      label="Dashboard"
                      labelIcon={<BarChart3 size={15} />}
                      href="/dashboard"
                    />
                    <UserButton.Action label="manageAccount" />
                  </UserButton.MenuItems>
                </UserButton>
              </SignedIn>
            </>
          )}
        </div>
      </nav>

      {showSignIn && (
        <div
          className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50"
          onClick={handleOverlayClick}
        >
          <SignIn
            signUpForceRedirectUrl="/fraud-detection"
            fallbackRedirectUrl="/fraud-detection"
          />
        </div>
      )}
    </>
  );
};

export default Header;
