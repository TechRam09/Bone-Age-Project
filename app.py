import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow import keras
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
model = None  # Lazy loading

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        model = load_model('model/bone_age_model.h5', compile=False)  # ✅ Load the model only once

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # ✅ Convert to Grayscale because model expects (224, 224, 1)
    img = Image.open(filepath).convert("L")  # 'L' = grayscale
    img = img.resize((224, 224))

    img_array = np.array(img).astype('float32') / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=-1)  # (224, 224, 1)
    img_array = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 1)
    
    gender = int(request.form.get('gender', 1))
    gender_input = np.array([[gender]])  # Example: 1 for Male, 0 for Female
    prediction = model.predict([img_array, gender_input], verbose=0)[0][0]
    
    return jsonify({'age': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
