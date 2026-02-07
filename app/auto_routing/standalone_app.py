"""
Standalone Flask app for testing the auto-routing service independently
Run this file directly to test the routing API without the main application
"""
from flask import Flask
from flask_cors import CORS
from routing import auto_routing_bp

def create_standalone_app():
    """Create a standalone Flask app for testing"""
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    # Register the auto-routing blueprint
    app.register_blueprint(auto_routing_bp, url_prefix="/api/auto-routing")
    
    # Add a root endpoint for testing
    @app.route("/")
    def index():
        return {
            "message": "Auto-Routing Service - Standalone Test Server",
            "version": "1.0.0",
            "endpoints": {
                "health": "GET /api/auto-routing/health",
                "optimize": "POST /api/auto-routing/optimize"
            },
            "status": "running"
        }
    
    return app

if __name__ == "__main__":
    app = create_standalone_app()
    print("=" * 70)
    print("🚀 AUTO-ROUTING SERVICE - STANDALONE TEST SERVER")
    print("=" * 70)
    print("\n📍 Server running at: http://localhost:5001")
    print("\n📋 Available endpoints:")
    print("   • GET  http://localhost:5001/")
    print("   • GET  http://localhost:5001/api/auto-routing/health")
    print("   • POST http://localhost:5001/api/auto-routing/optimize")
    print("\n💡 Use Postman or curl to test the API")
    print("   See POSTMAN_COLLECTION.json for example requests")
    print("\n" + "=" * 70 + "\n")
    
    # Run on port 5001 to avoid conflicts with main app
    app.run(debug=True, host="0.0.0.0", port=5001)
