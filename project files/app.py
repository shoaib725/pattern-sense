from flask import Flask, render_template, request
from flask import Flask, render_template, request
import tensorflow as tf
from keras.utils import load_img, img_to_array
import numpy as np
import os


# Initialize Flask app
app = Flask(__name__)

# Load model once at startup
model = tf.keras.models.load_model("fabric_model.h5")

# Define your class labels (make sure this matches your model training)
classes = ['floral', 'geometric', 'polka_dot', 'stripes']

# File upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    if 'file' not in request.files:
        return "No file uploaded!"
    file = request.files['file']
    if file.filename == '':
        return "No file selected!"

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load and preprocess image
    image = load_img(filepath, target_size=(128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]

    return render_template('results.html', prediction=predicted_class, img_path=filepath)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
