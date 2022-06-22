import numpy as np
import onnxruntime as ort
import cv2
import torch
import base64
import uuid
from data import get_valid_transforms
import warnings
warnings.filterwarnings('ignore')


def base64_to_image(base64_value):
    image_data = base64.b64decode(base64_value)
    file_name = f'{uuid.uuid4()}.jpg'
    with open(file_name, 'wb') as f:
        f.write(image_data)
    f.close()
    return file_name

def processing_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    transforms = get_valid_transforms()
    image = transforms(image=image)['image']
    image = torch.unsqueeze(image, 0)
    return image

def predict(image):
    ort_session = ort.InferenceSession('models/best_model.onnx')
    ort_inputs = {
        'modelInput': image.numpy()
    }
    ort_outs = ort_session.run(None, ort_inputs)
    pred = np.argmax(ort_outs)
    return pred

if __name__ == '__main__':
    labels = ['Cat', 'Dog']
    b64_value = ""
    image_path = base64_to_image(b64_value)
    image = processing_image(image_path)
    pred = predict(image)
    print(labels[pred])