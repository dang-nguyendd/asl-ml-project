import streamlit as st
# from google.cloud import aiplatform
# import tensorflow as tf
# import numpy as np
# from tensorflow import keras
# from keras.models import load_model
# from rag import *
import requests
import json
import base64
from oauth2client.client import GoogleCredentials
import os
import google.auth
import google.auth.transport.requests
import uuid
# Replace with your endpoint URL
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
CLASSES = ["Tomato___Bacterial_spot", 
"Tomato___Early_blight", 
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Tomato_mosaic_virus",
"Tomato___healthy"]

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service-account-file.json"

# Use the google.auth library to get the credentials and the access token
# credentials, project = google.auth.default()
# auth_req = google.auth.transport.requests.Request()
# credentials.refresh(auth_req)
# token = credentials.token

ENDPOINT="2952758267601747968"
PROJECT="853791352780"
INPUT_DATA_FILE="INPUT-JSON"
REGION = "us-central1"

# Cache model so the app doesn't need to load the model from the second prediction.
# @st.cache_resource(show_spinner=False)
# def load_and_cache_model():
#     model_path = "model_mb_v2_finetuned_5_reducelr.hdf5"
#     model = load_model(model_path)
#     return model

# def read_image(img_bytes):
#     img = tf.image.decode_jpeg(img_bytes, channels=IMG_CHANNELS)
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     return img






def b64encode(filename):
    with open(filename, "rb") as ifp:
        img_bytes = ifp.read()
        return base64.b64encode(img_bytes).decode()




def model_predict(save_path):
    token = (
      GoogleCredentials.get_application_default().get_access_token().access_token
    )
    headers = {"Authorization": "Bearer " + token}

    data = {
        "signature_name": "predict_base64",
        "instances": [{"img_bytes": {"b64": b64encode(save_path)}}],
    }

    api = "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/endpoints/{}:rawPredict".format(REGION, PROJECT, REGION, ENDPOINT)

    response = requests.post(api, json=data, headers=headers)
    json.loads(response.content)
    return response.content



def main():
  st.set_page_config(page_title='5-Flower Classifier', page_icon='🌿')

  st.title('Tomato Leaf Disease Classifier')

  st.markdown("This is a simple app to predict tomato disease base on your upload image.")

  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    # Generate a unique filename to avoid overwriting existing files
    unique_filename = f"uploaded_image_{uuid.uuid4().hex[:8]}.png"
    
    # Save the uploaded file locally
    save_path = os.path.join(".", unique_filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    # st.image(image.numpy(), caption='Uploaded Image', use_column_width=True)
    class_btn = st.button("Classify")
    if class_btn:
      with st.spinner('Model predicting....'):
        result = model_predict(save_path)
        st.write(result)
        # global model
        # model = load_and_cache_model()
        # prob, prediction = predict(model, image)
        # st.success(f"Prediction: {prediction} - {prob:.2%}")
        # recommend = ask_question(f"How to handle {prediction}?")
        # st.write(recommend)

if __name__ == "__main__":
  main()
