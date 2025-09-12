import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Upload, FileImage, AlertTriangle, CheckCircle, Download, BarChart3, Brain, Clock, Shield } from "lucide-react";
import { useDropzone } from "react-dropzone";
import { generateFraudReport } from "@/utils/pdfGenerator";

const RiskAssessment = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [modelSource, setModelSource] = useState(null);

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
    console.log('Analyze button clicked!', { selectedFile, loading });
    if (!selectedFile) {
      console.log('No file selected, returning early');
      return;
    }

    console.log('Starting analysis...');
    setLoading(true);
    try {
      // Convert file to base64
      const reader = new FileReader();
      reader.onload = async () => {
        console.log('File read successfully, converting to base64...');
        const base64 = reader.result.split(',')[1];
        
        // Use real ML API in development, Netlify function in production
        const apiUrl = import.meta.env.DEV 
          ? 'http://localhost:8001/predict-base64'  // Real ML API
          : '/.netlify/functions/predict-fraud';
        
        console.log('Making API call to:', apiUrl);
        
        // Show progress to user
        const startTime = Date.now();
        console.log('Starting ML analysis at:', new Date().toISOString());
        
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

        console.log('API Response status:', response.status, response.ok);
        if (!response.ok) {
          throw new Error(`API request failed: ${response.status}`);
        }

        const result = await response.json();
        const endTime = Date.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(1);
        console.log(`Analysis completed in ${processingTime}s`);
        console.log('API Response data:', result);
        
        if (result.success) {
          console.log('Setting prediction from result.prediction:', result.prediction);
          setPrediction(result.prediction);
          setModelSource(result.source || 'unknown');
        } else if (result.prediction) {
          // Handle direct ML API response format
          console.log('Setting prediction from direct result.prediction:', result.prediction);
          setPrediction(result.prediction);
          setModelSource(result.source || 'direct_api');
        } else {
          console.error('Invalid response format:', result);
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
      console.log('Analysis complete, setting loading to false');
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
      case 'VERY HIGH': return 'bg-red-700 text-white border border-red-400';
      case 'HIGH': return 'bg-orange-600 text-white border border-orange-400';
      case 'MODERATE': return 'bg-yellow-400 text-black border border-yellow-500';
      case 'LOW': return 'bg-blue-500 text-white border border-blue-400';
      case 'VERY LOW': return 'bg-green-500 text-white border border-green-400';
      default: return 'bg-gray-500 text-white border border-gray-400';
    }
  };

  return (
    <div className="container mx-auto py-8 px-4">
      {/* Enhanced Header */}
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-3 mb-6">
          <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl">
            <Brain className="h-8 w-8 text-white" />
          </div>
          <div className="text-left">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              AI Fraud Detection Analysis
            </h1>
            <p className="text-slate-600 text-lg">Advanced Neural Network • Real-time Analysis</p>
          </div>
        </div>
        <div className="max-w-2xl mx-auto">
          <p className="text-gray-600 mb-4">
            Upload vehicle damage images for instant AI-powered fraud detection analysis
          </p>
          <div className="flex items-center justify-center gap-6 text-sm text-slate-500">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span>91.4% Accuracy</span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-blue-500" />
              <span>Real-time Processing</span>
            </div>
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-purple-500" />
              <span>Secure Analysis</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Enhanced Upload Section */}
        <Card className="shadow-lg border-2">
          <CardHeader className="bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-800 dark:to-slate-900">
            <CardTitle className="flex items-center gap-3 text-lg">
              <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
                <Upload className="h-5 w-5 text-blue-600" />
              </div>
              Image Upload & Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 cursor-pointer ${
                isDragActive || dragActive
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 scale-105'
                  : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50/50'
              }`}
            >
              <input {...getInputProps()} />
              <div className="flex flex-col items-center gap-4">
                <div className={`p-4 rounded-full ${
                  isDragActive ? 'bg-blue-100 dark:bg-blue-900' : 'bg-gray-100 dark:bg-gray-800'
                }`}>
                  <FileImage className={`h-12 w-12 ${
                    isDragActive ? 'text-blue-600' : 'text-gray-400'
                  }`} />
                </div>
                <div>
                  <p className="text-lg font-medium text-slate-700 dark:text-slate-300">
                    {isDragActive ? '📁 Drop the image here' : '🖼️ Drag & drop an image here'}
                  </p>
                  <p className="text-sm text-gray-500">or click to select a file</p>
                  <p className="text-xs text-gray-400 mt-2">
                    Supports: JPG, PNG, WEBP (max 10MB)
                  </p>
                </div>
              </div>
            </div>

            {selectedFile && (
              <div className="mt-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-xl border border-green-200 dark:border-green-700">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-green-100 dark:bg-green-900 rounded-lg">
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    </div>
                    <div>
                      <p className="font-semibold text-green-800 dark:text-green-300">{selectedFile.name}</p>
                      <p className="text-sm text-green-600 dark:text-green-400">
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Ready for Analysis
                      </p>
                    </div>
                  </div>
                  <Button 
                    onClick={analyzeFraud} 
                    disabled={loading}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-6 py-2 rounded-lg shadow-lg font-semibold"
                  >
                    {loading ? (
                      <div className="flex items-center gap-2">
                        <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                        Analyzing...
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <Brain className="h-4 w-4" />
                        Analyze with AI
                      </div>
                    )}
                  </Button>
                </div>
              </div>
            )}

            {selectedFile && (
              <div className="mt-4">
                <div className="relative rounded-xl overflow-hidden shadow-lg">
                  <img
                    src={URL.createObjectURL(selectedFile)}
                    alt="Selected"
                    className="w-full h-64 object-cover"
                  />
                  <div className="absolute top-3 left-3 bg-black/70 text-white px-3 py-1 rounded-full text-sm font-medium">
                    Preview
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Enhanced Results Section */}
        <Card className="shadow-lg border-2">
          <CardHeader className="bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-800 dark:to-slate-900">
            <CardTitle className="flex items-center gap-3 text-lg">
              <div className="p-2 bg-purple-100 dark:bg-purple-900 rounded-lg">
                <BarChart3 className="h-5 w-5 text-purple-600" />
              </div>
              AI Analysis Results
              {prediction && (
                <div className="ml-auto">
                  <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                    prediction.prediction === 'FRAUD' 
                      ? 'bg-red-100 text-red-700' 
                      : 'bg-green-100 text-green-700'
                  }`}>
                    {prediction.prediction}
                  </div>
                </div>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            {loading && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-lg font-medium mb-2">🧠 AI is analyzing your image...</p>
                <p className="text-sm text-gray-600 mb-4">This may take 30-60 seconds for the first analysis</p>
                <div className="space-y-2 text-xs text-gray-500 max-w-md mx-auto">
                  <p>• Converting image to analysis format</p>
                  <p>• Running neural network detection</p>
                  <p>• Calculating fraud probability</p>
                  <p>• Generating recommendations</p>
                </div>
              </div>
            )}

            {prediction && !loading && (
              <div className="space-y-6">
                {/* Main Prediction Result Card */}
                <div className="bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-800 dark:to-slate-900 p-6 rounded-xl border">
                  <div className="text-center">
                    <div className={`inline-flex items-center gap-3 px-6 py-3 rounded-full text-white text-lg font-bold shadow-lg ${
                      prediction.prediction === 'FRAUD' 
                        ? 'bg-gradient-to-r from-red-500 to-red-600' 
                        : 'bg-gradient-to-r from-green-500 to-green-600'
                    }`}>
                      {prediction.prediction === 'FRAUD' ? (
                        <AlertTriangle className="h-6 w-6" />
                      ) : (
                        <CheckCircle className="h-6 w-6" />
                      )}
                      <span>{prediction.prediction}</span>
                    </div>
                    <div className="mt-4">
                      <p className="text-3xl font-bold text-slate-800 dark:text-white">{prediction.confidence}%</p>
                      <p className="text-slate-600 dark:text-slate-300">Confidence Score</p>
                    </div>
                    <div className={`inline-flex items-center gap-2 mt-3 px-4 py-2 rounded-full text-sm font-medium ${
                      prediction.riskLevel === 'VERY LOW' ? 'bg-green-100 text-green-800' :
                      prediction.riskLevel === 'LOW' ? 'bg-blue-100 text-blue-800' :
                      prediction.riskLevel === 'MODERATE' ? 'bg-yellow-100 text-yellow-800' :
                      prediction.riskLevel === 'HIGH' ? 'bg-orange-100 text-orange-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      <span>Risk Level: {prediction.riskLevel}</span>
                    </div>
                  </div>
                </div>

                {/* Detailed Analysis Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  
                  {/* Probability Breakdown */}
                  <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border shadow-sm">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                      <BarChart3 className="h-5 w-5 text-blue-600" />
                      Probability Analysis
                    </h3>
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm font-medium text-red-600">Fraud Probability</span>
                          <span className="text-lg font-bold text-red-600">{prediction.fraudProbability}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3">
                          <div 
                            className="bg-gradient-to-r from-red-400 to-red-600 h-3 rounded-full transition-all duration-1000" 
                            style={{width: `${prediction.fraudProbability}%`}}
                          ></div>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm font-medium text-green-600">Non-Fraud Probability</span>
                          <span className="text-lg font-bold text-green-600">{prediction.nonFraudProbability}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3">
                          <div 
                            className="bg-gradient-to-r from-green-400 to-green-600 h-3 rounded-full transition-all duration-1000" 
                            style={{width: `${prediction.nonFraudProbability}%`}}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* AI Model Performance Metrics */}
                  <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border shadow-sm">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                      <Brain className="h-5 w-5 text-purple-600" />
                      Model Performance
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                        <p className="text-2xl font-bold text-purple-600">91.4%</p>
                        <p className="text-xs text-purple-700 dark:text-purple-300">Accuracy</p>
                      </div>
                      <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                        <p className="text-2xl font-bold text-blue-600">87.9%</p>
                        <p className="text-xs text-blue-700 dark:text-blue-300">Precision</p>
                      </div>
                      <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                        <p className="text-2xl font-bold text-green-600">89.2%</p>
                        <p className="text-xs text-green-700 dark:text-green-300">Recall</p>
                      </div>
                      <div className="text-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                        <p className="text-2xl font-bold text-orange-600">88.5%</p>
                        <p className="text-xs text-orange-700 dark:text-orange-300">F1-Score</p>
                      </div>
                    </div>
                  </div>

                  {/* Processing Details */}
                  <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border shadow-sm">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                      <Clock className="h-5 w-5 text-indigo-600" />
                      Processing Details
                    </h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm text-slate-600 dark:text-slate-300">Processing Time</span>
                        <span className="font-semibold">{prediction.processingTime}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-slate-600 dark:text-slate-300">Model Architecture</span>
                        <span className="font-semibold">EfficientNet-B1</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-slate-600 dark:text-slate-300">Input Resolution</span>
                        <span className="font-semibold">224x224</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-slate-600 dark:text-slate-300">Analysis Date</span>
                        <span className="font-semibold">{new Date().toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>

                  {/* Risk Assessment */}
                  <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border shadow-sm">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                      <Shield className="h-5 w-5 text-amber-600" />
                      Risk Assessment
                    </h3>
                    <div className="space-y-3">
                      <div className={`p-3 rounded-lg ${
                        prediction.riskLevel === 'VERY LOW' ? 'bg-green-50 border border-green-200' :
                        prediction.riskLevel === 'LOW' ? 'bg-blue-50 border border-blue-200' :
                        prediction.riskLevel === 'MODERATE' ? 'bg-yellow-50 border border-yellow-200' :
                        prediction.riskLevel === 'HIGH' ? 'bg-orange-50 border border-orange-200' :
                        'bg-red-50 border border-red-200'
                      }`}>
                        <p className="font-semibold text-sm">Risk Category</p>
                        <p className="text-lg font-bold">{prediction.riskLevel}</p>
                      </div>
                      <div className="p-3 bg-slate-50 dark:bg-slate-700 rounded-lg">
                        <p className="font-semibold text-sm mb-1">Recommended Action</p>
                        <p className="text-sm">{prediction.recommendedAction}</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Model Source Indicator */}
                <div className={`p-4 rounded-xl border-2 shadow-sm ${
                  modelSource === 'real_model' 
                    ? 'bg-green-50 border-green-200 text-green-800' 
                    : 'bg-orange-50 border-orange-200 text-orange-800'
                }`}>
                  <div className="flex items-center gap-3 text-sm font-medium">
                    {modelSource === 'real_model' ? (
                      <>
                        <CheckCircle className="h-5 w-5" />
                        <div>
                          <p className="font-bold">✅ Real AI Model Active</p>
                          <p className="text-xs opacity-75">PyTorch EfficientNet-B1 (91.4% accuracy) • Source: {modelSource}</p>
                        </div>
                      </>
                    ) : (
                      <>
                        <AlertTriangle className="h-5 w-5" />
                        <div>
                          <p className="font-bold">⚠️ Fallback Mode</p>
                          <p className="text-xs opacity-75">Mock prediction (ML API unavailable) • Source: {modelSource}</p>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {/* Enhanced Generate Report Button */}
                <div className="mt-6 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl border border-indigo-200 dark:border-indigo-700">
                  <div className="text-center mb-3">
                    <h4 className="font-semibold text-indigo-800 dark:text-indigo-300">Generate Professional Report</h4>
                    <p className="text-sm text-indigo-600 dark:text-indigo-400">Download detailed analysis in PDF format</p>
                  </div>
                  <Button 
                    onClick={generateReport} 
                    className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white py-3 rounded-lg shadow-lg font-semibold"
                    size="lg"
                  >
                    <Download className="h-5 w-5 mr-2" />
                    Generate PDF Report
                  </Button>
                </div>
              </div>
            )}

            {!prediction && !loading && (
              <div className="text-center py-12">
                <div className="p-6 bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-800 dark:to-slate-900 rounded-xl border-2 border-dashed border-slate-300 dark:border-slate-600">
                  <div className="p-4 bg-slate-100 dark:bg-slate-700 rounded-full w-fit mx-auto mb-4">
                    <BarChart3 className="h-12 w-12 text-slate-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-2">Ready for Analysis</h3>
                  <p className="text-slate-500 dark:text-slate-400">Upload a vehicle damage image to start AI-powered fraud detection</p>
                  <div className="flex items-center justify-center gap-4 mt-4 text-xs text-slate-400">
                    <span>• Advanced ML Analysis</span>
                    <span>• Instant Results</span>
                    <span>• Professional Reports</span>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default RiskAssessment;
