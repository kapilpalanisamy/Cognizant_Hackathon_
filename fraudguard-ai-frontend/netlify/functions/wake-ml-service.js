// Keep-alive function to wake up the ML service
const ML_API_URL = process.env.ML_API_URL || 'https://fraudguard-ai-backend.onrender.com';

exports.handler = async (event, context) => {
  // Handle CORS
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, OPTIONS'
      }
    };
  }

  try {
    console.log('Attempting to wake up ML service:', ML_API_URL);
    
    const response = await fetch(`${ML_API_URL}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });

    if (response.ok) {
      const result = await response.text();
      return {
        statusCode: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          success: true,
          message: 'ML service is awake',
          status: response.status,
          response: result,
          timestamp: new Date().toISOString()
        })
      };
    } else {
      return {
        statusCode: response.status,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          success: false,
          message: 'ML service responded but not healthy',
          status: response.status,
          timestamp: new Date().toISOString()
        })
      };
    }
  } catch (error) {
    console.error('Wake-up error:', error);
    
    return {
      statusCode: 503,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        success: false,
        error: 'Service unavailable',
        message: error.message,
        timestamp: new Date().toISOString()
      })
    };
  }
};