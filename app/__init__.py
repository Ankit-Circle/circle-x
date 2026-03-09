from flask import Flask, jsonify
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    # Import Blueprints
    from app.image_enhancement import image_enhancement_bp
    from app.image_convert import image_convert_bp

    # Pricing depends on external services and env vars.
    # Import it lazily so the core app (e.g. image conversion)
    # can run without full pricing configuration.
    pricing_bp = None
    pricing_db_bp = None
    try:
        from app.pricing import pricing_bp as _pricing_bp, pricing_db_bp as _pricing_db_bp  # type: ignore
        pricing_bp = _pricing_bp
        pricing_db_bp = _pricing_db_bp
    except Exception:
        pricing_bp = None
        pricing_db_bp = None

    # Register with `/api/` prefix
    app.register_blueprint(image_enhancement_bp, url_prefix="/api/enhance")
    if pricing_bp is not None:
        app.register_blueprint(pricing_bp, url_prefix="/api/pricing")
    if pricing_db_bp is not None:
        app.register_blueprint(pricing_db_bp, url_prefix="/api/pricing-db")
    app.register_blueprint(image_convert_bp, url_prefix="/api/convert")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "service": "circle-x",
            "blueprints": list(app.blueprints.keys())
        }), 200
    return app
