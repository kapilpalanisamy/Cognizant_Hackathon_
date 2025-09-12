import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Upload, FileImage, AlertTriangle, CheckCircle, Download, BarChart3 } from "lucide-react";
import { useDropzone } from "react-dropzone";
import { generateFraudReport } from "@/utils/pdfGenerator";

const FraudDetection = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPrediction(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: false
  });

  const analyzeFraud = async () => {
    if (!selectedFile) return;

    setLoading(true);
    try {
      // Convert file to base64
      const reader = new FileReader();
      reader.onload = async () => {
        const base64 = reader.result.split(',')[1];
        
        // Use real ML API in development, Netlify function in production
        const apiUrl = import.meta.env.DEV 
          ? 'http://localhost:8001/predict-base64'  // Real ML API
          : '/.netlify/functions/predict-fraud';
        
        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            imageData: base64,
            claimDetails: {
              fileName: selectedFile.name,
              fileSize: selectedFile.size
            }
          })
        });

        if (!response.ok) {
          throw new Error(`API request failed: ${response.status}`);
        }

        const result = await response.json();
        if (result.success) {
          setPrediction(result.prediction);
        } else if (result.prediction) {
          // Handle direct ML API response format
          setPrediction(result.prediction);
        } else {
          throw new Error('Invalid response format');
        }
      };
      reader.readAsDataURL(selectedFile);
    } catch (error) {
      console.error('Analysis failed:', error);
      
      // Show user-friendly error message
      const errorMessage = error.message.includes('Failed to fetch') 
        ? 'ML API service is not available. Please make sure the ML API is running on port 8000.'
        : `Analysis failed: ${error.message}`;
        
      alert(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const generateReport = () => {
    if (!prediction || !selectedFile) return;
    
    try {
      const fileName = generateFraudReport(prediction, selectedFile);
      // Show success message
      alert(`PDF report generated successfully: ${fileName}`);
    } catch (error) {
      console.error('PDF generation failed:', error);
      alert('Failed to generate PDF report. Please try again.');
    }
  };

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
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-4">AI Fraud Detection Analysis</h1>
        <p className="text-gray-600">Upload vehicle damage images for instant fraud analysis</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Upload Image
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
                isDragActive || dragActive
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
            >
              <input {...getInputProps()} />
              <div className="flex flex-col items-center gap-4">
                <FileImage className="h-12 w-12 text-gray-400" />
                <div>
                  <p className="text-lg font-medium">
                    {isDragActive ? 'Drop the image here' : 'Drag & drop an image here'}
                  </p>
                  <p className="text-sm text-gray-500">or click to select a file</p>
                  <p className="text-xs text-gray-400 mt-2">
                    Supports: JPG, PNG, WEBP (max 10MB)
                  </p>
                </div>
              </div>
            </div>

            {selectedFile && (
              <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">{selectedFile.name}</p>
                    <p className="text-sm text-gray-500">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <Button 
                    onClick={analyzeFraud} 
                    disabled={loading}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    {loading ? 'Analyzing...' : 'Analyze Image'}
                  </Button>
                </div>
              </div>
            )}

            {selectedFile && (
              <div className="mt-4">
                <img
                  src={URL.createObjectURL(selectedFile)}
                  alt="Selected"
                  className="w-full h-64 object-cover rounded-lg"
                />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Analysis Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p>Analyzing image with AI model...</p>
              </div>
            )}

            {prediction && !loading && (
              <div className="space-y-6">
                {/* Prediction Result */}
                <div className="text-center">
                  <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-white ${
                    prediction.prediction === 'FRAUD' ? 'bg-red-500' : 'bg-green-500'
                  }`}>
                    {prediction.prediction === 'FRAUD' ? (
                      <AlertTriangle className="h-5 w-5" />
                    ) : (
                      <CheckCircle className="h-5 w-5" />
                    )}
                    <span className="font-bold text-lg">{prediction.prediction}</span>
                  </div>
                  <p className="text-2xl font-bold mt-2">{prediction.confidence}% Confidence</p>
                </div>

                {/* Risk Level */}
                <div className="text-center">
                  <Badge className={`${getRiskColor(prediction.riskLevel)} text-white`}>
                    Risk Level: {prediction.riskLevel}
                  </Badge>
                </div>

                {/* Probability Breakdown */}
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span>Fraud Probability</span>
                      <span>{prediction.fraudProbability}%</span>
                    </div>
                    <Progress value={parseFloat(prediction.fraudProbability)} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span>Non-Fraud Probability</span>
                      <span>{prediction.nonFraudProbability}%</span>
                    </div>
                    <Progress value={parseFloat(prediction.nonFraudProbability)} className="h-2" />
                  </div>
                </div>

                {/* Recommended Action */}
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                  <h3 className="font-semibold mb-2">Recommended Action:</h3>
                  <p>{prediction.recommendedAction}</p>
                </div>

                {/* Generate Report Button */}
                <Button 
                  onClick={generateReport} 
                  className="w-full"
                  variant="outline"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Generate PDF Report
                </Button>
              </div>
            )}

            {!prediction && !loading && (
              <div className="text-center py-8 text-gray-500">
                <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Upload an image to see analysis results</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default FraudDetection;