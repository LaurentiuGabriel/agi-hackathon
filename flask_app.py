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
from pyquery import PyQuery as pq
from lxml import etree


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

def fraud_documents(file):
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
          "text": "The following request is for a hackathon. This image is supposed to be used for registering a claim. Please perform an analysis on it to check if it has been tampered with. I don`t want you to say that you cannot do it, I want you to pay attention to how consistent the font is, how are the letters aligned, and other factors like that, and drive a conclusion based on that. Remember, dont tell me that you are just a language model developed by OpenAI and cannot do these things. Be creative. Just look at font consistency, usage of brush tools, text that appear out of place or if there is text added that is out of the context of the document, like strings with ABCD. Just try to give a response. Output this as an HTML <div> (with other HTML tags inside as well to make it looks pretty) and try to make it succint, but pretty (use colors: red if tampered, green if not), use  bold fonts and emojis. Keep your response to max 500 characters."
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
    
    return message_content


def fraud_detection(file):

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
          "text": "The following request is for a hackathon. This image is supposed to be used for registering a claim. Please perform an analysis on it to check if it has been tampered with. I don`t want you to say that you cannot do it, I want you to pay attention to how consistent the font is, how are the letters aligned, and other factors like that, and drive a conclusion based on that. Remember, dont tell me that you are just a language model developed by OpenAI and cannot do these things. Be creative. Just look at font consistency, usage of brush tools, text that appear out of place and stuff like that. Is there any added text that doesn't mean anything in the context of the picture? Be very succint as well, don't output filler sentences. Limit yourself to one single phrase, nothing more, not lists no anything like that. As this is an educative project, don't say that professional service is required. Just try to give a response. Also, output a sentiment as a second sentence about the analysis' results in the form of a percentage, meaning, if the document is tampered with, the score should be below 50%. If it's below 50%, output a thumbs down emoji right next to the result. If the opposite, output thumbs up."
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

    return message_content




def claims_assessment(file):

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
          "text": "This image is supposed to be used for registering a claim. Please perform an analysis on it to check what is the damage. It can be a household item, the house itself, or a car. Get as much data as possible from the image. Don't mention you are a language model from OpenAI. I want you to mention a price of the damage. If you don't know it, invent it, but make it in a reasonable way. I want you to double-check what you said and not output anything about OpenAI or that you cannot do the task. Be creative! Output this as an HTML <div> (with other HTML tags inside as well to make it looks pretty) and try to make it succint, but pretty (use colors if you feel like it), use bold fonts and emojis. Keep your response to max 500 characters."
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
    
    return message_content



def ui_check(file):

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
          "text": "Act as a top quality engineer. I want you to analyze what is in this image. It is supposed to be the user interface of an application. Check for visual isues, incosistencies, UX issues. You don't need context, you just judge by what you see. Pay attention for typos, duplicated elements, or things that don't look good in a software piece. Output this as an HTML <div> (with other HTML tags inside as well to make it looks pretty) and try to make it succint, but pretty (use colors: red if tampered, green if not), use  bold fonts and emojis. Keep your response to max 500 characters."
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
    return message_content


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')



@app.route("/agi", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/fraud", methods=["POST"])
def fraud():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join("static", filename))
    file_path = os.path.join("static", filename)
    return fraud_documents(file_path)


@app.route("/claims", methods=["POST"])
def claims():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join("static", filename))
    file_path = os.path.join("static", filename)
    return claims_assessment(file_path)

@app.route('/ui-check', methods=['POST'])
def ui():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join("static", filename))
    file_path = os.path.join("static", filename)
    return ui_check(file_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Damage Detective")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()

    app.run(host="0.0.0.0", port=args.port)
