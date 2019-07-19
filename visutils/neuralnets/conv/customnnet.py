# -*- encoding: utf-8 -*-
# customnet.py
'''
Here you can implement and edit your own architecture.
To load it, pass `custom` as arg value in --architecture.
'''

import keras.backend
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense


class CustomNet:
    '''
    class
    '''
    @staticmethod
    def build(width, height, depth, classes):
        '''
        '''
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if keras.backend.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = 1
        
        # first block
        model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model