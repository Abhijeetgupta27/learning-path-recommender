import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.models.recommend import generate_learning_path

# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)
CORS(app)


# -----------------------------
# HEALTH CHECK (useful in prod)
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running"}), 200


# -----------------------------
# RECOMMENDATION ENDPOINT
# -----------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()

        # ✅ Input validation
        if not data or "interests" not in data:
            logging.warning("Invalid request received")
            return jsonify({"error": "Invalid input. Provide 'interests' list."}), 400

        interests = data["interests"]

        # Ensure it's a list
        if not isinstance(interests, list):
            logging.warning("Interests is not a list")
            return jsonify({"error": "'interests' must be a list"}), 400

        logging.info(f"Request received: {interests}")

        # ✅ Call model
        results = generate_learning_path(interests)

        logging.info(f"Returned {len(results)} recommendations")

        return jsonify(results.to_dict(orient="records"))

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)