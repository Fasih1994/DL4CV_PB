from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Flatten
import keras.backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):

        model = Sequential()
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            chanDim = 1
        else:
            input_shape = (height, width, depth)
            chanDim = -1

        # block 1
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # block 2
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # block 3 fully connected block
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        # softmax block
        model.add((Dense(classes)))
        model.add(Activation('softmax'))

        return model