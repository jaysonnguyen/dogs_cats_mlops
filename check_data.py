import os
import cv2


train_dir = 'train'
valid_dir = 'valid'

train_images = os.listdir(train_dir)
valid_images = os.listdir(valid_dir)

for image in train_images:
    image_path = os.path.join(train_dir, image)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    try:
        if img.shape[0] < 128 or img.shape[1] < 128:
            print(image)
            os.remove(image_path)
    except:
        print('Error:', image)
        os.remove(image_path)