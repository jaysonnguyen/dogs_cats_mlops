import os
import random

source = 'kagglecatsanddogs_5340/PetImages/Dog'

cat_files = os.listdir('kagglecatsanddogs_5340/PetImages/Cat')
dog_files = os.listdir('kagglecatsanddogs_5340/PetImages/Dog')

random.shuffle(dog_files)
train_files = dog_files[:10000]
valid_files = dog_files[10000:]
print(train_files[:5])
print(len(train_files))
print(len(valid_files))

def rename(file, type):
    src = f'{source}/{file}'
    dst = f'dataset/{type}/Dog_{file}'
    print(src, dst)
    os.rename(src, dst)

for file in train_files:
    rename(file, 'train')
for file in valid_files:
    rename(file, 'valid')
