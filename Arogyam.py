import sys
import os
import glob
import re

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np

from flask import Flask


app = Flask(__name__)

model = load_model("models/Skin_Disease.h5")


def prediction(img_path, model):
    img = image.load_img(
        img_path, target_size=(224, 224)
    )  # Adjust target_size based on your model's input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    label = ["Actinic keratosis", "Atopic Dermatitis", "Dermatofibroma", "Melanoma"]
    predicted_class_index = np.argmax(predictions)
    predicted_label = label[predicted_class_index]
    return predicted_label


@app.route("/", methods=["GET"])
def index():
    # Main page
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Get the file from post request
        f = request.files["file"]

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = prediction(file_path, model)  # Convert to string
        return preds
    return None


if __name__ == "__main__":
    app.run(debug=True)
