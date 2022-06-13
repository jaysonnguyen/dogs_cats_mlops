import torch
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class DogAndCat(Dataset):
    def __init__(self, data_dir, transforms):
        super(DogAndCat, self).__init__()

        self.data_dir = data_dir
        self.images = os.listdir(data_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        label = image.split('_')[0]
        image = cv2.imread(f'{self.data_dir}/{image}', cv2.COLOR_BGR2RGB)
        image /= 255.0
        classes = ['Dog', 'Cat']
        lb = LabelEncoder()
        lb.fit_transform(classes)
        label = lb.transform(label)
        label = torch.tensor(label)
        if self.transforms:
            image = self.transforms(image)
        return image, label


def get_train_transforms():
    return A.Compose([
        A.Resize(256, 256, p=1.0),
        ToTensorV2(p=1.0)
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(256, 256, p=1.0),
        ToTensorV2(p=1.0)
    ])

