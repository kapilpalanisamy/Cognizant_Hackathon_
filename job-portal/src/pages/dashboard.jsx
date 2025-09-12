import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BarChart3, Shield, TrendingUp, Clock, AlertTriangle, CheckCircle } from "lucide-react";

const Dashboard = () => {
  // Mock data - this would come from your backend in a real app
  const stats = {
    totalAnalyses: 1247,
    fraudDetected: 189,
    accuracy: 91.4,
    avgProcessingTime: 2.3
  };

  const recentAnalyses = [
    {
      id: 1,
      fileName: "claim_001.jpg",
      prediction: "NON-FRAUD",
      confidence: 89.2,
      timestamp: "2 minutes ago",
      riskLevel: "LOW"
    },
    {
      id: 2,
      fileName: "claim_002.jpg", 
      prediction: "FRAUD",
      confidence: 94.7,
      timestamp: "15 minutes ago",
      riskLevel: "VERY HIGH"
    },
    {
      id: 3,
      fileName: "claim_003.jpg",
      prediction: "NON-FRAUD",
      confidence: 76.3,
      timestamp: "1 hour ago",
      riskLevel: "MODERATE"
    }
  ];

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'VERY HIGH': return 'bg-red-500';
      case 'HIGH': return 'bg-orange-500';
      case 'MODERATE': return 'bg-yellow-500';
      case 'LOW': return 'bg-blue-500';
      case 'VERY LOW': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">Fraud Detection Dashboard</h1>
        <p className="text-gray-600">Monitor AI analysis performance and recent activity</p>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Analyses</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalAnalyses.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">+12% from last month</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Fraud Detected</CardTitle>
            <AlertTriangle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{stats.fraudDetected}</div>
            <p className="text-xs text-muted-foreground">
              {((stats.fraudDetected / stats.totalAnalyses) * 100).toFixed(1)}% of total
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Accuracy</CardTitle>
            <TrendingUp className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{stats.accuracy}%</div>
            <p className="text-xs text-muted-foreground">Enterprise-grade performance</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Processing</CardTitle>
            <Clock className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{stats.avgProcessingTime}s</div>
            <p className="text-xs text-muted-foreground">Lightning fast analysis</p>
          </CardContent>
        </Card>
      </div>

      {/* Model Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <Card>
          <CardHeader>
            <CardTitle>Model Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span>Precision</span>
                <span className="font-bold text-blue-600">87.9%</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Recall</span>
                <span className="font-bold text-green-600">86.0%</span>
              </div>
              <div className="flex justify-between items-center">
                <span>F1-Score</span>
                <span className="font-bold text-purple-600">86.9%</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Overall Accuracy</span>
                <span className="font-bold text-orange-600">91.4%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Risk Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span>Very High Risk</span>
                <Badge className="bg-red-500 text-white">8.2%</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span>High Risk</span>
                <Badge className="bg-orange-500 text-white">12.6%</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span>Moderate Risk</span>
                <Badge className="bg-yellow-500 text-white">19.4%</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span>Low Risk</span>
                <Badge className="bg-blue-500 text-white">35.1%</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span>Very Low Risk</span>
                <Badge className="bg-green-500 text-white">24.7%</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Analyses */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Analyses</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {recentAnalyses.map((analysis) => (
              <div key={analysis.id} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    {analysis.prediction === 'FRAUD' ? (
                      <AlertTriangle className="h-5 w-5 text-red-500" />
                    ) : (
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    )}
                    <span className="font-medium">{analysis.fileName}</span>
                  </div>
                  <Badge 
                    variant={analysis.prediction === 'FRAUD' ? 'destructive' : 'default'}
                  >
                    {analysis.prediction}
                  </Badge>
                  <Badge className={`${getRiskColor(analysis.riskLevel)} text-white`}>
                    {analysis.riskLevel}
                  </Badge>
                </div>
                <div className="flex items-center space-x-4 text-sm text-gray-500">
                  <span>{analysis.confidence}% confidence</span>
                  <span>{analysis.timestamp}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;