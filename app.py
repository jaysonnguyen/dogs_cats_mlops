from fastapi import FastAPI
import inference_onnx

app = FastAPI(title="Dogs And Cats MLOps")

@app.get('/')
async def home():
    return "<h2>Home"

@app.get('/predict')
async def get_prediction(image_path: str):
    labels = ['Cat', 'Dog']
    img = inference_onnx.processing_image(image_path)
    pred = inference_onnx.predict(img)
    result = {
        'result': labels[pred],
    }
    return result