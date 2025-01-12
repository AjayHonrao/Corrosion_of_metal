from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import os

# Load the saved model
model = tf.keras.models.load_model('corrosion_classification_model.keras')

# Initialize the Flask app
app = Flask(__name__)

# Function to predict the class of the image
def predict_image(model, img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    return "no-Corrosion" if prediction[0] > 0.5 else "Corrosion"

# Define the route
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            result = "No file uploaded"
        else:
            file = request.files['file']
            if file.filename == '':
                result = "No file selected"
            else:
                img_path = os.path.join('uploads', file.filename)
                file.save(img_path)

                result = predict_image(model, img_path)
                os.remove(img_path)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
