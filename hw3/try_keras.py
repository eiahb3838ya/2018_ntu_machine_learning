
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop


# In[3]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']
def reconstruct(pix_str, size=(48,48)):
    pix_arr = np.array(pix_str.split(' '),dtype=float)
    return pix_arr
def load_data(filepath='train.csv'):
    data = pd.read_csv(filepath)
    # print df.tail()
    # print df.Usage.value_counts()
#     df = df[df.Usage == usage]
#     frames = []
    
#     for _class in classes:
#         class_df = df[df['emotion'] == emotion[_class]]
#         frames.append(class_df)
#     data = pd.concat(frames, axis=0)
#     rows = random.sample(data.index, int(len(data)*sample_split))
#     data = data.ix[rows]
#     print ('{} set for {}: {}'.format(usage, classes, data.shape))
    data['feature'] = data.feature.apply(lambda x: reconstruct(x))
    x = np.array([mat for mat in data.feature]).reshape(-1,48,48)# (n_samples, img_width, img_height)
    X_train = x.reshape(-1, 1, x.shape[1], x.shape[2])
    # y_train=to_categorical(data['label'])
    print("X_train",X_train.shape)
    return (X_train)


# In[4]:


emo = ['Angry', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']
# reconstruct(dataset.feature[0])
# X_train, y_train =load_data()
X_train=np.load("raw_data/X_train.npy")
y_train=np.load("raw_data/y_train.npy")

# In[10]:


#params:
batch_size = 100
epochs = 13

X_train=X_train[:28700]
y_train=y_train[:28700]
# setup info:
print ('X_train shape: ', X_train.shape) # (n_sample, 1, 48, 48)
print ('y_train shape: ', y_train.shape) # (n_sample, n_categories)
print( '  img size: ', X_train.shape[2], X_train.shape[3])
print ('batch size: ', batch_size)
print( '  nb_epoch: ', epochs)


# In[8]:


model = Sequential()
model.add(Conv2D(32,(3, 3), padding='same', activation='relu',
                        input_shape=(1, X_train.shape[2], X_train.shape[3])))
model.add(Conv2D(32,(3, 3), padding='same', activation='relu',))
model.add(Conv2D(32,(3, 3), padding='same', activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Conv2D(64,(3, 3), padding='same', activation='relu',))
model.add(Conv2D(64,(3, 3), padding='same', activation='relu',))
model.add(Conv2D(64,(3, 3), padding='same', activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))


model.add(Conv2D(128,(3, 3), padding='same', activation='relu',))
model.add(Conv2D(128,(3, 3), padding='same', activation='relu',))
model.add(Conv2D(128,(3, 3), padding='same', activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','metrics.categorical_accuracy'])


# In[ ]:


# print 'Training....'
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
          validation_split=0.3, shuffle=True, verbose=1)



loss_and_metrics = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=1)
print ('Done!')
print ('Loss: ', loss_and_metrics[0])
print (' Acc: ', loss_and_metrics[1])

X_test =load_data("raw_data/test.csv")
answer=model.predict_classes(X_test)
np.save("answer/answer.npy",answer)