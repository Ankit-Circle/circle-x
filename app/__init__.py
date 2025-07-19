from flask import Flask

def create_app():
    app = Flask(__name__)

    # Import Blueprints
    from app.image_enhancement import image_enhancement_bp
    from app.pricing import pricing_bp
    from app.pricing import pricing_db_bp

    # Register with `/api/` prefix
    app.register_blueprint(image_enhancement_bp, url_prefix="/api/enhance")
    app.register_blueprint(pricing_bp, url_prefix="/api/pricing")
    app.register_blueprint(pricing_db_bp, url_prefix="/api/pricing-db")

    return app
