import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('corrosion_classification_model.keras')

def predict_image(model, img_path):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(224, 224))  # Changed to load_img
    img_array = img_to_array(img)  # Changed to img_to_array
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make the prediction
    prediction = model.predict(img_array)

    # Return the result
    return "no-Corrosion" if prediction[0] > 0.5 else "Corrosion"

def predict_images_in_folder(model, folder_path):
    predictions = []
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image (you can adjust the extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            result = predict_image(model, img_path)
            predictions.append((filename, result))
    
    return predictions

# Example usage
folder_path = r'C:\Users\Ajay\OneDrive\Desktop\corrosion\coroosion'
predictions = predict_images_in_folder(model, folder_path)

# Print predictions
for filename, result in predictions:
    print(f"{filename}: {result}")
