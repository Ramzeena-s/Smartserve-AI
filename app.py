
from flask import Flask, request, render_template_string
from datetime import datetime
import pandas as pd
import joblib
import os

app = Flask(__name__)

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


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    status = None
    status_class = None
    error = None
    tips = []
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            date = pd.to_datetime(request.form["date"], errors="raise")
            meals = int(request.form["meals_served"])
            staff = int(request.form["kitchen_staff"])
            temperature = float(request.form["temperature_C"])
            humidity = float(request.form["humidity_percent"])
            past_waste = float(request.form["past_waste_kg"])

            if meals <= 0 or staff <= 0:
                raise ValueError("Meals served and kitchen staff must be greater than zero.")

            if not -20 <= temperature <= 60:
                raise ValueError("Temperature must be between -20 and 60 °C.")

            if not 0 <= humidity <= 100:
                raise ValueError("Humidity must be between 0 and 100%.")

            if past_waste < 0:
                raise ValueError("Previous food waste cannot be negative.")

            # Prepare features in the same order and names used in training
            input_data = pd.DataFrame([{
                "meals_served": meals,
                "kitchen_staff": staff,
                "temperature_C": temperature,
                "humidity_percent": humidity,
                "day_of_week": date.day_name(),
                "special_event": request.form["special_event"],
                "past_waste_kg": past_waste,
                "staff_experience": match_category(
                    request.form["staff_experience"], 2
                ),
                "waste_category": match_category(
                    request.form["waste_category"], 3
                ),
                "year": date.year,
                "month": date.month,
                "day_number": date.day
            }])

            result = float(model.predict(input_data)[0])
            prediction = round(max(0, result), 2)
            status, status_class = get_waste_status(prediction)

            if status == "Low":
                tips = [
                    "Continue tracking daily food waste.",
                    "Keep preparing food according to observed demand."
                ]
            elif status == "Moderate":
                tips = [
                    "Review previous meal counts before preparing food.",
                    "Prepare food in smaller batches when practical.",
                    "Track which food categories generate the most waste."
                ]
            else:
                tips = [
                    "Compare food prepared with meals actually served.",
                    "Consider smaller preparation batches and timely replenishment.",
                    "Review storage, portion sizes, and leftover records."
                ]

            recent_predictions.insert(0, {
                "date": date.strftime("%d %b %Y"),
                "meals": meals,
                "waste": prediction,
                "status": status,
                "status_class": status_class
            })
            del recent_predictions[5:]

        except (ValueError, TypeError, KeyError) as exc:
            error = f"Please check your inputs: {exc}"

    total_predictions = len(recent_predictions)
    average_waste = (
        round(sum(item["waste"] for item in recent_predictions)
              / total_predictions, 2)
        if total_predictions else 0
    )

    with open(os.path.join(app.root_path, "index.html"),
              "r", encoding="utf-8") as file:
        html = file.read()

    return render_template_string(
        html,
        prediction=prediction,
        status=status,
        status_class=status_class,
        error=error,
        tips=tips,
        form_data=form_data,
        recent_predictions=recent_predictions,
        total_predictions=total_predictions,
        average_waste=average_waste,
        now=datetime.now().strftime("%Y-%m-%d")
    )


if __name__ == "__main__":
    app.run(debug=True)
