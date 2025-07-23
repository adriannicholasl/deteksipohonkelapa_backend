import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 128
GRID_SIZE = 4
PATCH_SIZE = IMG_SIZE // GRID_SIZE

def preprocess_whole_image(image):
    img = load_img(image, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_patch_image(image):
    img = load_img(image, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    patches = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            patch = arr[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE, :]
            patches.append(patch)
    return np.expand_dims(np.array(patches), axis=0)
