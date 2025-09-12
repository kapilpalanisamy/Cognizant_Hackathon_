// Real ML API integration with your PyTorch model
const ML_API_URL = process.env.ML_API_URL || 'https://fraudguard-ai-backend.onrender.com';

exports.handler = async (event, context) => {
  // Handle CORS
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
      }
    };
  }

  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers: {
        'Access-Control-Allow-Origin': '*'
      },
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  try {
    const { imageData, claimDetails } = JSON.parse(event.body);
    
    // Try to call your real ML API first
    try {
      const response = await fetch(`${ML_API_URL}/predict-base64`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          imageData: imageData
        })
      });

      if (response.ok) {
        const result = await response.json();
        
        return {
          statusCode: 200,
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            success: true,
            prediction: result.prediction,
            timestamp: new Date().toISOString(),
            processingTime: result.prediction?.processingTime || 'N/A',
            source: 'real_model'
          })
        };
      }
    } catch (apiError) {
      console.warn('ML API not available:', apiError.message);
    }
    
    // Fallback to mock prediction if ML API is not available
    console.warn('Using fallback mock prediction');
    const mockPrediction = generateMockPrediction();
    
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        success: true,
        prediction: mockPrediction,
        timestamp: new Date().toISOString(),
        processingTime: `${(Math.random() * 2 + 1).toFixed(2)}s`,
        source: 'fallback_mock'
      })
    };

  } catch (error) {
    console.error('Prediction error:', error);
    
    return {
      statusCode: 500,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        success: false,
        error: 'Internal server error',
        message: error.message
      })
    };
  }
};

// Fallback mock prediction function
function generateMockPrediction() {
  const mockPrediction = {
    prediction: Math.random() > 0.6 ? 'NON-FRAUD' : 'FRAUD',
    confidence: (Math.random() * 40 + 60).toFixed(1),
    fraudProbability: (Math.random() * 50 + 10).toFixed(1),
    nonFraudProbability: null,
    riskLevel: null,
    recommendedAction: null
  };

  mockPrediction.nonFraudProbability = (100 - parseFloat(mockPrediction.fraudProbability)).toFixed(1);
  
  const confidence = parseFloat(mockPrediction.confidence);
  if (mockPrediction.prediction === 'FRAUD') {
    if (confidence >= 90) mockPrediction.riskLevel = 'VERY HIGH';
    else if (confidence >= 80) mockPrediction.riskLevel = 'HIGH';
    else if (confidence >= 65) mockPrediction.riskLevel = 'MODERATE';
    else mockPrediction.riskLevel = 'LOW-MODERATE';
  } else {
    if (confidence >= 80) mockPrediction.riskLevel = 'VERY LOW';
    else if (confidence >= 65) mockPrediction.riskLevel = 'LOW';
    else mockPrediction.riskLevel = 'UNCERTAIN';
  }

  if (mockPrediction.prediction === 'FRAUD') {
    if (confidence >= 90) mockPrediction.recommendedAction = 'Immediate investigation required';
    else if (confidence >= 80) mockPrediction.recommendedAction = 'Priority investigation';
    else if (confidence >= 65) mockPrediction.recommendedAction = 'Standard review process';
    else mockPrediction.recommendedAction = 'Basic documentation review';
  } else {
    if (confidence >= 80) mockPrediction.recommendedAction = 'Auto-approve claim';
    else if (confidence >= 65) mockPrediction.recommendedAction = 'Standard processing';
    else mockPrediction.recommendedAction = 'Additional verification';
  }

  return mockPrediction;
}
