from model import get_unet, img_rows, img_cols
from generator import DataGenerator
import json
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="9" 

weights_path = 'best.hdf5'
model = get_unet()
model.load_weights(weights_path)

with open('best-idx', 'r') as f:
    predict = json.loads(f.read())
for filename in predict:
    filename = filename.split('.')[0]
    image = imread('train/image/' + filename + '.jpg')
    image = resize(image, (img_rows, img_cols), order=1,
                       anti_aliasing=True) # bi-linear
    mask = model.predict(np.array([image]))

    with open('predict/' + filename + '.npy', 'w') as f:
        f.write(json.dumps(mask.tolist()))