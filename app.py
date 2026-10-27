import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Facial Emotion Recognition")
st.write("Upload a face image to predict its facial expression.")

def load_model():
    return tf.keras.models.load_model("best_emotion_model.keras")

model = load_model()
class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"]

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=250)

    image = image.resize((48, 48))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 48, 48, 1)

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = np.argmax(predictions)

    st.subheader("Prediction")
    st.write("Predicted expression:", class_names[predicted_index])


