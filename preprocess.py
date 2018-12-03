
# coding: utf-8

# In[11]:


import os
import random
import numpy as np
from skimage.transform import *
from skimage.io import imsave, imread


# In[33]:


def aug_rotate(image, deg):
    return rotate(image, deg)

def aug_flipud(image):
    return np.flipud(image)

def aug_fliplr(image):
    return np.fliplr(image)

def aug_translatex(image, x):
    return warp(image, EuclideanTransform(translation=(x, 0)))

def aug_translatey(image, y):
    return warp(image, EuclideanTransform(translation=(0, y)))


# In[40]:


images = os.listdir('data/image')

augmentations = [
    aug_rotate,
    aug_flipud,
    aug_fliplr,
    aug_translatex,
    aug_translatey
]

translate_range = list(range(-51, 0)) + list(range(1, 52))

count = 0
for filename in images:
    filename = filename.split('.')[0]
    image = imread('data/image/' + filename + '.jpg')
    mask = imread('data/mask/' + filename + '_segmentation.png')
    
    imsave('train/image/' + filename + '.jpg', image)
    imsave('train/mask/' + filename + '_segmentation.png', mask)
    
    samples = random.sample(range(5), 4)
    rotate_deg = random.randint(45, 315)
    translatex = random.choice(translate_range)
    translatey = random.choice(translate_range)
    for (i, aug_idx) in enumerate(samples):    

        if aug_idx == 1 or aug_idx == 2:
            aug_image = augmentations[aug_idx](image)
            aug_mask = augmentations[aug_idx](mask)

        if aug_idx == 0:
            aug_image = augmentations[aug_idx](image, rotate_deg)
            aug_mask = augmentations[aug_idx](mask, rotate_deg)
        if aug_idx == 3:
            aug_image = augmentations[aug_idx](image, translatex)
            aug_mask = augmentations[aug_idx](mask, translatex)    
        if aug_idx == 4:
            aug_image = augmentations[aug_idx](image, translatey)
            aug_mask = augmentations[aug_idx](mask, translatey)

        imsave('train/image/' + filename + '_' + str(i) + '.jpg', aug_image)
        imsave('train/mask/' + filename + '_' + str(i) + '_segmentation.png', aug_mask)
    count += 1
    if count % 100 == 0:
        print(count)
    

