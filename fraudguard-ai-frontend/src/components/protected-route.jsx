/* eslint-disable react/prop-types */
import { Navigate, useLocation } from "react-router-dom";
import { useUser } from "@clerk/clerk-react";

const ProtectedRoute = ({ children }) => {
  // Development mode - bypass authentication
  const isDevelopment = !import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
  
  if (isDevelopment) {
    return children;
  }

  const { isSignedIn, isLoaded } = useUser();

  // Redirect to auth page if not signed in
  if (isLoaded && !isSignedIn) {
    return <Navigate to="/" />;
  }

  // Show loading or return children if authenticated
  return children;
};

export default ProtectedRoute;
