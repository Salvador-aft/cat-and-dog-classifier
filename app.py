from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os

app = FastAPI()

model = tf.keras.models.load_model('cat_dog_classifier.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return "dog" if prediction > 0.5 else "cat"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as file:
        return file.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return {"error": "File must be an image"}

    img_path = "temp_image.jpg"
    with open(img_path, "wb") as buffer:
        buffer.write(await file.read())

    result = predict_image(img_path)

    os.remove(img_path)

    return {"result": result}