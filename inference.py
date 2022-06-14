import torch
import numpy as np
import cv2
import torchvision
from data import get_valid_transforms


def processing_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    return image


def predict(image):
    model = torchvision.models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=2, bias=True)
    model.load_state_dict(torch.load('models/best_model.pt'))
    model.eval()
    output = model(image)
    _, pred = torch.max(output, dim=1)
    pred = pred.item()
    return pred

if __name__ == '__main__':
    labels = {
        0: 'Cat',
        1: 'Dog'
    }
    image = processing_image('dataset/train/Dog_10833.jpg')
    transforms = get_valid_transforms()
    image = transforms(image=image)['image']
    image = torch.unsqueeze(image, 0)
    output = predict(image)
    if output in list(labels.keys()):
        print(labels[output])