from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import sys

app = FastAPI()

def load(path: str):
    return load_model(path)

def predict_digit(model, data_point):
    data_point = np.array(data_point).reshape(1, 28* 28)
    prediction = model.predict(data_point)
    return str(np.argmax(prediction))

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img)
    data_point = img_array.flatten() / 255.0  # Flatten and normalize
    model_path = sys.argv[1]  # Get model path from command line argument
    model = load(model_path)
    digit = predict_digit(model, data_point)
    return {"digit": digit}

if __name__ == "__main__":
    import uvicorn
    model_path = sys.argv[1]  # Get model path from command line argument
    model = load(model_path)
    uvicorn.run(app, host="0.0.0.0", port=8000)
