import argparse
import os
from geopy.geocoders import Nominatim
import torch
from flask import Flask, render_template, request, redirect
from keras.preprocessing import image
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from flask import Flask, render_template, request, redirect, jsonify
from werkzeug.utils import secure_filename
from OpenSSL import SSL
import base64
import requests

context = SSL.Context(SSL.SSLv23_METHOD)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
EXIF_DATE_FORMAT = "%Y:%m:%d %H:%M:%S"
ERROR_LEVEL_ANALYSIS = "Not done"
HISTOGRAM_ANALYSIS = "Not done"
METADATA_CHECK = "Not done"
GRADIENT_LUMINANCE_ANALYSIS = "Not done"
COPY_MOVE_DETECTION_ANALYSIS = "Not done"

app = Flask(__name__)

api_key = os.environ["OPENAI"]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def chat_with_gpt_fraud_detection(file):

    base64_image = encode_image(file)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "user",
            "content": [
        {
          "type": "text",
          "text": "This image is supposed to be used for registering a claim. Please perform an analysis on it to check if it has been tampered with. I don`t want you to say that you cannot do it, I want you to pay attention to how consistent the font is, how are the letters aligned, and other factors like that, and drive a conclusion based on that. Remember, dont tell me that you are just a language model and cannot do these things."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()
    message_content = response_json['choices'][0]['message']['content']
    lines = message_content.split('\n')

    return lines


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')




@app.route("/formdata", methods=["POST"])
def post_formdata():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file']

    filename = secure_filename(file.filename)

    
    file.save(os.path.join("static", filename))

        
    file_path = os.path.join("static", filename)
    
    return chat_with_gpt_fraud_detection(file_path)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        uploaded_file = request.files["file"]
        if uploaded_file.filename == '':
            return redirect(request.url)
            
       
        
        return redirect("static/images/image0.jpg")

    return render_template("index.html")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file']
    
    return chat_with_gpt_fraud_detection(file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Damage Detective")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()

    app.run(host="0.0.0.0", port=args.port)
