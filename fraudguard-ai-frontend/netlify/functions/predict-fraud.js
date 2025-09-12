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
    
    // Try to call your real ML API first with extended timeout for Render cold starts
    try {
      console.log('Attempting to connect to ML API:', ML_API_URL);
      
      // Render free tier can take up to 60+ seconds for cold start
      // We'll use a longer timeout but still within Netlify's limits
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('API timeout after 25 seconds')), 25000) // 25 second timeout for cold starts
      );
      
      const fetchPromise = fetch(`${ML_API_URL}/predict-base64`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          imageData: imageData
        })
      });

      console.log('Waiting for ML API response (this may take time for first request on Render free tier)...');
      const response = await Promise.race([fetchPromise, timeoutPromise]);

      if (response.ok) {
        const result = await response.json();
        console.log('ML API response received successfully:', result);
        
        return {
          statusCode: 200,
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            success: true,
            prediction: result.prediction || result, // Handle different response formats
            timestamp: new Date().toISOString(),
            processingTime: result.prediction?.processingTime || result.processingTime || 'N/A',
            source: 'real_model'
          })
        };
      } else {
        console.warn(`ML API returned status: ${response.status}`);
        throw new Error(`API returned ${response.status}: ${response.statusText}`);
      }
    } catch (apiError) {
      console.error('ML API not available:', apiError.message);
      
      // If it's a timeout, show a specific message
      if (apiError.message.includes('timeout')) {
        console.error('ML API timed out - likely a cold start delay on Render free tier');
      }
      
      // Return error instead of fallback
      return {
        statusCode: 503,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          success: false,
          error: 'ML API Unavailable',
          message: 'The fraud detection ML API is currently unavailable. Please try again in a few moments.',
          details: apiError.message,
          timestamp: new Date().toISOString()
        })
      };
    }

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
