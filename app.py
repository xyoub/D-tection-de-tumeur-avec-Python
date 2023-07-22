# Import necessary libraries
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('BrainTumor10Epochs2.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    if classNo == 0:
        return "Pas de présence de tumeurs"
    elif classNo == 1:
        return "Présence de tumeur"


def getResult(img):
    # Read and process the image
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    
    # Make predictions using the loaded model
    result = model.predict(input_img)
    predicted_class = np.argmax(result[0])
    return predicted_class


@app.route('/', methods=['GET'])
def index():
    # Serve the index.html template
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Retrieve the uploaded file
        f = request.files['file']

        # Save the file to the 'uploads' directory
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Get the prediction for the uploaded image
        value = getResult(file_path)
        result = get_className(value)

        # Return the predicted class as a response
        return result

    return None


if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
