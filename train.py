import os
import json
from keras.callbacks import *
from sklearn.model_selection import KFold
from generator import DataGenerator
from model import get_unet, img_cols, img_rows

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="9" 



data_dir = '/mnt/dfs/reggie/isic/train/'

images = os.listdir(os.path.join(data_dir, 'image'))
images = [image.split('.')[0] for image in images]


kf = KFold(n_splits=5, shuffle=True)

model = get_unet()

checkpoint_fp = 'weights-12011708.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint_cb = ModelCheckpoint(checkpoint_fp, verbose=1, save_best_only=True)

early_stopping = EarlyStopping(monitor='val_jaccard', mode='max', verbose=1, patience=5)

lr_scheduler = LearningRateScheduler(lambda epoch: 1e-4 if epoch < 9 else 1e-5, verbose=1)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

history = LossHistory()

cb_list = [checkpoint_cb, early_stopping, history, lr_scheduler]

for train_idx, valid_idx in kf.split(images):
    train = [images[idx] for idx in train_idx]
    valid = [images[idx] for idx in valid_idx]
    train_generator = DataGenerator(train, batch_size=64, dim=(img_rows, img_cols))
    valid_generator = DataGenerator(valid, batch_size=64, dim=(img_rows, img_cols))
    model.fit_generator(epochs=100, generator=train_generator,
                    validation_data=valid_generator,
                    verbose=1,
                    callbacks=cb_list)
    with open("valid_idx-12011708", 'w') as f:
        f.write(json.dumps(valid));
    break
