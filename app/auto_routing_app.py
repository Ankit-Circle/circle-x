from flask import Flask, jsonify
from flask_cors import CORS

def create_auto_routing_app():
    app = Flask(__name__)
    CORS(app)

    from app.auto_routing import auto_routing_bp
    app.register_blueprint(
        auto_routing_bp,
        url_prefix="/api/auto-routing"
    )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "service": "auto-routing"
        }), 200

    return app
