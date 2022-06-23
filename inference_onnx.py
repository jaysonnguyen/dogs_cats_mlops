import numpy as np
import onnxruntime as ort
import cv2
import torch
import base64
from data import get_valid_transforms
import warnings
warnings.filterwarnings('ignore')


def processing_image(base64_value):
    image_data = base64.b64decode(base64_value)
    np_image = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    transforms = get_valid_transforms()
    image = transforms(image=image)['image']
    image = torch.unsqueeze(image, 0)
    return image

def predict(image):
    labels = ['Cat', 'Dog']
    ort_session = ort.InferenceSession('models/best_model.onnx')
    ort_inputs = {
        'modelInput': image.numpy()
    }
    ort_outs = ort_session.run(None, ort_inputs)
    pred = np.argmax(ort_outs)
    return labels[pred]

if __name__ == '__main__':
    b64_value = ""
    image = processing_image(b64_value)
    pred = predict(image)
    print(pred)