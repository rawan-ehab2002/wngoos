
import gradio as gr
import tensorflow as tf
import joblib
from PIL import Image
import numpy as np

# Image size used in training
IMG_SIZE = (224, 224)

# Load your trained model
model = tf.keras.models.load_model("clothing_classifier_model.keras")  # or the relative path on HF

# Load label encoder
le = joblib.load("category_encoder.pkl")

# Image preprocessing function (same as training)
def preprocess_image(img: Image.Image):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = img_array.astype(np.float32)
    return np.expand_dims(img_array, axis=0)

def predict(image):
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    class_index = np.argmax(prediction)
    class_label = le.inverse_transform([class_index])[0]
    return class_label

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Fashion Product Image"),
    outputs=gr.Label(label="Predicted Category"),
    title="Fashion Category Classifier",
    description="Upload a fashion image to predict its category"
)

if __name__ == "__main__":
    interface.launch()
