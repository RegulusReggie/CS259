
# coding: utf-8

# In[67]:


import os
import json
import numpy as np
from skimage.transform import resize
from skimage.io import imread

masks = os.listdir('predict')
predict = np.zeros(shape=(0, 2))
gts = np.zeros(shape=(0))

for filename in masks:
    with open('predict/' + filename, 'r') as f:
        mask = json.loads(f.read())
        mask = np.array(mask)
        mask = mask.flatten()
        mask = np.array([[x, 1-x] for x in mask])
        predict = np.concatenate((predict, mask))
    maskgt = imread('train/mask/' + filename.split('.')[0] + '_segmentation.png')
    maskgt = resize(maskgt, (96, 96), order=0, anti_aliasing=True) # nearest neighbor
    maskgt[maskgt > 0] = 1
    maskgt = maskgt.flatten()
#     maskgt = np.array([[x] for x in maskgt])
    gts = np.concatenate((gts, maskgt))

print(predict.shape)
print(gts.shape)


# In[68]:


np.unique(gts)


# In[80]:


from sklearn.metrics import roc_curve, auc
# print(predict[:, 0].shape)
fpr, tpr, thresholds = roc_curve(gts, predict[:, 0])
rocauc = auc(fpr, tpr)


# In[96]:


idx = -1
dist = 999

for i in range(len(fpr)):
    d = fpr[i] ** 2 + (1 - tpr[i]) ** 2
    if d < dist:
        dist = d
        idx = i


# In[98]:


print(idx)
print(fpr[idx], tpr[idx], thresholds[idx])
print(fpr[0:5], tpr[0:5], thresholds[0:5])


# In[99]:


thres = thresholds[idx]


# In[124]:


from skimage.io import imsave

count = 0
for filename in masks:
    if len(filename.split('_')) > 2: continue
    with open('predict/' + filename, 'r') as f:
        mask = json.loads(f.read())
        mask = np.array(mask)
        mask[mask > thres] = 255
        mask[mask <= thres] = 0
        mask = np.reshape(mask, (96, 96))
#         raw = imread('/home/reggie/cs259/data/raw/' + filename.split('.')[0] + '.jpg')
#         print(raw.shape)
#         mask = resize(mask, (raw.shape[0], raw.shape[1]), order=0, anti_aliasing=True) # nearest neighbor
        imsave(filename.split('.')[0] + '.png', mask.astype('int'))
        
    count += 1
    if count == 5: break


# In[127]:


import matplotlib.pyplot as plt

plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % rocauc)
plt.title('ROC curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('roc.png')
plt.show()
plt.close()

