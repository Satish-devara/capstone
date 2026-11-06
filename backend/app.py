"""
app.py
Flask application factory — entry point for the backend API.
"""

from flask import Flask, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from config import get_config

# Global db instance (imported by models and routes)
db = SQLAlchemy()


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(get_config())

    # Initialize extensions
    db.init_app(app)
    CORS(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGIN"]}})

    # Register route blueprints
    from routes.predict_routes import predict_bp
    from routes.history_routes import history_bp
    from routes.model_routes import model_bp

    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(history_bp, url_prefix="/api")
    app.register_blueprint(model_bp, url_prefix="/api")

    # Health check endpoint
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "diabetes-prediction-api", "version": "1.0.0"})

    # Global error handlers
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Resource not found."}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error."}), 500

    # Create database tables
    with app.app_context():
        db.create_all()

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)

