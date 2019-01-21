from common_functions import *
import pandas as pd 
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,Conv2D,AveragePooling2D,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import LeakyReLU,PReLU
from keras.optimizers import SGD, RMSprop,Adam
import os
from sklearn.model_selection import train_test_split
# set vars


batch_size = 200
epochs = 8000

X_train=np.load("raw_data/X_train.npy")
y_train=np.load("raw_data/y_train.npy")

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
# X_train,y_train=load_data("raw_data/train.csv")
# np.save("raw_data/X_train.npy",X_train)
# np.save("raw_data/y_train.npy",y_train)
# X_train=X_train[:28700]
# y_train=y_train[:28700]
# setup info:
print ('X_train shape: ', X_train.shape) # (n_sample, 1, 48, 48)
print ('y_train shape: ', y_train.shape) # (n_sample, n_categories)
print( '  img size: ', X_train.shape[2], X_train.shape[3])
print ('batch size: ', batch_size)
print( '  nb_epoch: ', epochs)



#define model

from keras.applications.vgg16 import VGG16
model = VGG16(weights=None, include_top=True)



# 從頂部移出一層
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
# 加一層，只辨識10類
from keras.layers import Dense
num_classes=7
x=Dense(num_classes, activation='softmax')(model.output)
# 重新建立模型結構
model=Model(model.input,x)
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy',])

#apply ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True
    ) 
# set callback
early_stopping = EarlyStopping(monitor='val_loss', patience=21, verbose=2)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.35,verbose=1, patience=8, mode='auto',min_lr=0.000001,min_delta=0.00005)

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
          validation_split=0.1, shuffle=True, verbose=1,callbacks=[early_stopping,reduce_lr])

# datagen.fit(X_train)
# history=model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),validation_data=(X_val,y_val),
#                     steps_per_epoch=len(X_train)//(batch_size), epochs=epochs,verbose=1,callbacks=[early_stopping,reduce_lr])
loss_and_metrics = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=1)
print ('Done!')
# print ('Loss: ', loss_and_metrics[0])
# print (' Acc: ', loss_and_metrics[1])

draw_learning_line(history)

X_test =load_X_data("raw_data/test.csv")
answer=model.predict_classes(X_test)
np.save("answer/answer.npy",answer)
answer_to_csv()