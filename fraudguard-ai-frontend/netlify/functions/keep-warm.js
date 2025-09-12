// Keep the Render backend warm to reduce cold start times
const ML_API_URL = process.env.ML_API_URL || 'https://fraudguard-ai-backend.onrender.com';

exports.handler = async (event, context) => {
  try {
    const response = await fetch(`${ML_API_URL}/health`, {
      method: 'GET',
      timeout: 5000
    });
    
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        success: response.ok,
        timestamp: new Date().toISOString(),
        status: 'Backend keep-warm ping sent'
      })
    };
  } catch (error) {
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        success: false,
        error: error.message,
        timestamp: new Date().toISOString(),
        status: 'Keep-warm failed but this is non-critical'
      })
    };
  }
};