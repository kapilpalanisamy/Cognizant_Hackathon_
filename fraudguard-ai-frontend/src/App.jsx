import { RouterProvider, createBrowserRouter } from "react-router-dom";

import AppLayout from "./layouts/app-layout";
import ProtectedRoute from "./components/protected-route";
import { ThemeProvider } from "./components/theme-provider";

import AuthPage from "./pages/auth";
import RiskAssessment from "./pages/fraud-detection";

import "./App.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <AuthPage />,
  },
  {
    element: <AppLayout />,
    children: [
      {
        path: "/risk-assessment",
        element: (
          <ProtectedRoute>
            <RiskAssessment />
          </ProtectedRoute>
        ),
      },
    ],
  },
]);

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <RouterProvider router={router} />
    </ThemeProvider>
  );
}

export default App;
