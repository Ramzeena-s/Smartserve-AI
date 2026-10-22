from flask import Flask, request, render_template_string, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "https://ramzeena-s.github.io"
        }
    }
)

# Load the trained machine learning model
MODEL_PATH = os.path.join(app.root_path, "food_waste_model.pkl")
model = joblib.load(MODEL_PATH)

# Store recent predictions while the server is running
recent_predictions = []

# Demo thresholds; adjust these to suit your canteen
LOW_WASTE_LIMIT = 10
MODERATE_WASTE_LIMIT = 25


def get_waste_status(waste):
    if waste < LOW_WASTE_LIMIT:
        return "Low", "low"
    elif waste < MODERATE_WASTE_LIMIT:
        return "Moderate", "moderate"
    return "High", "high"


def match_category(value, column_index):
    """Match the user's selection to a category learned by the model."""
    try:
        encoder = (
            model.named_steps["preprocessor"]
            .named_transformers_["categorical"]
            .named_steps["encoder"]
        )
        known_values = encoder.categories_[column_index]

        for known in known_values:
            if str(known).strip().casefold() == str(value).strip().casefold():
                return known
    except (AttributeError, IndexError, KeyError):
        pass

    return value


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "SmartServe AI API is running.",
        "prediction_endpoint": "/api/predict",
        "method": "POST"
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}

    try:
        date = pd.to_datetime(data["date"], errors="raise")
        meals = int(data["meals_served"])
        staff = int(data["kitchen_staff"])
        temperature = float(data["temperature_C"])
        humidity = float(data["humidity_percent"])
        past_waste = float(data["past_waste_kg"])

        if meals <= 0 or staff <= 0:
            raise ValueError(
                "Meals and kitchen staff must be greater than zero."
            )

        if not -20 <= temperature <= 60:
            raise ValueError("Temperature must be between -20 and 60 °C.")

        if not 0 <= humidity <= 100:
            raise ValueError("Humidity must be between 0 and 100%.")

        if past_waste < 0:
            raise ValueError("Previous waste cannot be negative.")

        special_event = str(data["special_event"])
        if special_event not in ("Yes", "No"):
            raise ValueError("Choose Yes or No for special event.")

        staff_experience = match_category(
            data["staff_experience"], 2
        )
        waste_category = match_category(
            data["waste_category"], 3
        )

        input_data = pd.DataFrame([{
            "meals_served": meals,
            "kitchen_staff": staff,
            "temperature_C": temperature,
            "humidity_percent": humidity,
            "day_of_week": date.day_name(),
            "special_event": special_event,
            "past_waste_kg": past_waste,
            "staff_experience": staff_experience,
            "waste_category": waste_category,
            "year": date.year,
            "month": date.month,
            "day_number": date.day
        }])

        prediction = round(
            max(0, float(model.predict(input_data)[0])), 2
        )
        status, status_class = get_waste_status(prediction)

        if status == "Low":
            tips = [
                "Continue tracking daily food waste.",
                "Prepare food according to observed demand."
            ]
        elif status == "Moderate":
            tips = [
                "Review previous meal counts.",
                "Prepare food in smaller batches when practical.",
                "Track waste by food category."
            ]
        else:
            tips = [
                "Compare food prepared with meals served.",
                "Consider smaller preparation batches.",
                "Review storage and portion sizes."
            ]

        recent_predictions.insert(0, {
            "date": date.strftime("%d %b %Y"),
            "meals": meals,
            "waste": prediction,
            "status": status,
            "status_class": status_class
        })
        del recent_predictions[5:]

        average_waste = round(
            sum(item["waste"] for item in recent_predictions)
            / len(recent_predictions), 2
        )

        return jsonify({
            "success": True,
            "prediction": prediction,
            "status": status,
            "status_class": status_class,
            "tips": tips,
            "total_predictions": len(recent_predictions),
            "average_waste": average_waste,
            "recent_predictions": recent_predictions
        })

    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({
            "success": False,
            "error": f"Please check your inputs: {exc}"
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
