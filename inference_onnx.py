import numpy as np
import onnxruntime as ort
import cv2
import torch
from data import get_valid_transforms
import warnings
warnings.filterwarnings('ignore')


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
    image = processing_image('Cat_1558.jpg')
    pred = predict(image)
    print(labels[pred])