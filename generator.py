import numpy as np
import keras
from skimage.transform import resize
from skimage.io import imread

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(128, 128), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        idx_end = min(len(self.indexes), (index + 1) * self.batch_size)
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : idx_end]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            train_image = imread('/mnt/dfs/reggie/isic/train/image/' + ID + '.jpg')
            train_image = resize(train_image, self.dim, order=1,
                       anti_aliasing=True) # bi-linear

            # Store sample
            X[i,] = train_image

            train_mask = imread('/mnt/dfs/reggie/isic/train/mask/' + ID + '_segmentation.png')
            # train_mask = np.true_divide(train_mask, 255)
            train_mask = resize(train_mask, self.dim + (1,), order=0, anti_aliasing=True) # nearest neighbor

            # Store class
            y[i,] = train_mask

        return X, y
