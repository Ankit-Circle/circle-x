from flask import Flask, jsonify
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    # Import Blueprints
    from app.image_enhancement import image_enhancement_bp
    from app.pricing import pricing_bp
    from app.pricing import pricing_db_bp

    # Register with `/api/` prefix
    app.register_blueprint(image_enhancement_bp, url_prefix="/api/enhance")
    app.register_blueprint(pricing_bp, url_prefix="/api/pricing")
    app.register_blueprint(pricing_db_bp, url_prefix="/api/pricing-db")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "service": "circle-x",
            "blueprints": list(app.blueprints.keys())
        }), 200
    return app
