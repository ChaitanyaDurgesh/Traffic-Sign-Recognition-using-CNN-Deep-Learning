from __future__ import division, print_function
import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'model.h5'
LABELS_PATH = 'labels.csv'

# Load the model
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')

# Load class labels
def load_labels(labels_path):
    df = pd.read_csv(labels_path)
    class_labels = {row['ClassId']: row['Name'] for _, row in df.iterrows()}
    return class_labels

class_labels = load_labels(LABELS_PATH)
print('Labels loaded.')

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    return class_labels.get(classNo, "Unknown")

def model_predict(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(32, 32))
        img = np.asarray(img)
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=1)[0]  # get the index of the class with the highest probability
        preds = getClassName(classIndex)
        return preds
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return str(e)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        try:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            preds = model_predict(file_path, model)
            result = preds
            return jsonify(result=result)
        except Exception as e:
            print(f"Error during file upload and prediction: {e}")
            return jsonify(result=str(e))
    return jsonify(result="No file uploaded")

if __name__ == '__main__':
    app.run(port=5001, debug=True)
