import os
import cv2
import torch
import tempfile
from flask import Flask, render_template, request, send_file, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("D:/success/model/best.pt")

# Temporary file storage
temp_image_path = None

# Calorie information (all default per 100g)
calorie_info = {
    'Apple': 52,         # per 100g
    'Chapathi': 68,      # per 100g
    'Chicken Gravy': 150,# per 100g
    'Fries': 312,        # per 100g
    'Idli': 130,         # per 100g
    'Pizza': 266,        # per 100g
    'Rice': 130,         # per 100g
    'Soda': 42,          # per 100g (approx for 330ml)
    'Tomato': 18,        # per 100g
    'Vada': 290,         # per 100g
    'banana': 89,        # per 100g
    'burger': 295        # per 100g
}

@app.route("/", methods=["GET", "POST"])
def upload_and_detect():
    global temp_image_path

    detected_foods = {}

    if request.method == "POST":
        file = request.files["file"]
        if file:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            file.save(temp_file.name)

            image = cv2.imread(temp_file.name)
            image_height, image_width, _ = image.shape

            results = model(image)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    label = result.names[int(box.cls[0].item())]
                    base_calories = calorie_info.get(label, 0)

                    if label in detected_foods:
                        detected_foods[label]['base_calories'] += round(base_calories, 2)
                        detected_foods[label]['grams'] += 100
                    else:
                        detected_foods[label] = {
                            'base_calories': round(base_calories, 2),
                            'grams': 100
                        }

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{label} {base_calories:.2f} kcal", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            cv2.imwrite(temp_image_path, image)

    return render_template("index.html", detected_foods=detected_foods)

@app.route("/processed_image")
def serve_image():
    global temp_image_path
    if temp_image_path and os.path.exists(temp_image_path):
        return send_file(temp_image_path, mimetype="image/jpeg")
    return "No image processed yet", 404

@app.route("/update_calories", methods=["POST"])
def update_calories():
    data = request.json
    updated_foods = {}

    for item in data:
        label = item['label']
        grams = item['grams']
        base_calories = calorie_info.get(label, 0)
        calories = round((base_calories / 100) * grams, 2)
        updated_foods[label] = {
            'calories': calories,
            'grams': grams
        }
    return jsonify(updated_foods)

if __name__ == "__main__":
    app.run(debug=True)