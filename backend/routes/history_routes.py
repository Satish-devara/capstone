"""
routes/history_routes.py
GET  /api/history        — paginated prediction history
GET  /api/history/<id>   — single prediction record
DELETE /api/history/<id> — delete a prediction record
"""

from flask import Blueprint, request, jsonify
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.db_models import PredictionHistory
from app import db

history_bp = Blueprint("history", __name__)


@history_bp.route("/history", methods=["GET"])
def get_history():
    """
    Endpoint: GET /api/history?page=1&limit=10

    Returns a paginated list of past prediction records, newest first.

    Response (200):
        {
            "predictions": [ { ...record... }, ... ],
            "total": 42,
            "page": 1,
            "pages": 5,
            "limit": 10
        }
    """
    page  = request.args.get("page",  1,  type=int)
    limit = request.args.get("limit", 10, type=int)
    limit = min(limit, 100)   # Hard cap — never return more than 100 at once

    pagination = (
        PredictionHistory.query
        .order_by(PredictionHistory.timestamp.desc())
        .paginate(page=page, per_page=limit, error_out=False)
    )

    return jsonify({
        "predictions": [p.to_dict() for p in pagination.items],
        "total":  pagination.total,
        "page":   page,
        "pages":  pagination.pages,
        "limit":  limit,
    }), 200


@history_bp.route("/history/<int:record_id>", methods=["GET"])
def get_single(record_id: int):
    """
    Endpoint: GET /api/history/<id>

    Returns a single prediction record by ID.
    """
    record = PredictionHistory.query.get_or_404(record_id)
    return jsonify(record.to_dict()), 200


@history_bp.route("/history/<int:record_id>", methods=["DELETE"])
def delete_prediction(record_id: int):
    """
    Endpoint: DELETE /api/history/<id>

    Deletes a prediction record from the database.
    """
    record = PredictionHistory.query.get_or_404(record_id)
    db.session.delete(record)
    db.session.commit()
    return jsonify({"message": f"Record {record_id} deleted successfully."}), 200


@history_bp.route("/history", methods=["DELETE"])
def clear_history():
    """
    Endpoint: DELETE /api/history

    Deletes ALL prediction history. Use with caution.
    """
    count = PredictionHistory.query.delete()
    db.session.commit()
    return jsonify({"message": f"Deleted {count} records."}), 200

