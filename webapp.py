from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
from transformers import OPTForCausalLM, AutoTokenizer

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained skin cancer model
model = tf.keras.models.load_model('skin_cancer_detection.h5')
Categories = {'Benign': 'Not Harmful', 'Malignant': 'Harmful'}

# Load Facebook OPT-1.3B
opt_model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (175, 175))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_data = np.array(image).reshape(-1, 175, 175, 3)
    return image_data

# Function to generate explanation using Facebook OPT without the prompt question
def generate_disease_explanation(disease):
    prompt = f"{disease} skin cancer is a condition where abnormal skin cells grow uncontrollably. It is considered {Categories[disease]}. A brief explanation:"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = opt_model.generate(**inputs, max_length=100)
    explanation = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove any remaining references to the prompt
    explanation = explanation.replace(prompt, "").strip()
    
    return explanation

# Home Route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html", prediction="No file selected.", image=None)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess and predict
        image_data = preprocess_image(file_path)
        pred = model.predict(image_data)
        prediction = "Malignant" if int(pred[0][0]) == 1 else "Benign"

        # Generate explanation using Facebook OPT
        disease_explanation = generate_disease_explanation(prediction)
        harmful_status = Categories[prediction]

        return render_template("index.html", prediction=prediction, image=file_path, explanation=disease_explanation, harmful=harmful_status)

    return render_template("index.html", prediction=None, explanation=None, harmful=None)

# Run Flask App on Port 8100
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8100, debug=True)
