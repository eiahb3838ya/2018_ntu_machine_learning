import pandas as pd 
import numpy as np
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D,Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop
import matplotlib.pyplot as plt



emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']
def reconstruct(pix_str, size=(48,48)):
    pix_arr = np.array(pix_str.split(' '),dtype=float)
    return pix_arr

def load_data(filepath='train.csv'):
    data = pd.read_csv(filepath)
    data['feature'] = data.feature.apply(lambda x: reconstruct(x))
    x = np.array([mat for mat in data.feature]).reshape(-1,48,48)# (n_samples, img_width, img_height)
    X_train = x.reshape(-1,x.shape[1], x.shape[2],1)
    y_train=to_categorical(data['label'])
    print("X_train",X_train.shape,"y_train",y_train.shape)
    return (X_train,y_train)

def load_X_data(filepath='train.csv'):
    data = pd.read_csv(filepath)
    data['feature'] = data.feature.apply(lambda x: reconstruct(x))
    x = np.array([mat for mat in data.feature]).reshape(-1,48,48)# (n_samples, img_width, img_height)
    X_train = x.reshape(-1, x.shape[1], x.shape[2],1)
    print("shape of X:",X_train.shape)
    return (X_train)




def draw_learning_line(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def get_answer_num():
    with open("answer/answer_number.txt","r") as p:
        NUM=p.read()
        
    NUM=int(NUM)+1
    with open("answer/answer_number.txt","w")as p:
        p.write(str(NUM))
      
    return(NUM)

def answer_to_csv(): 
    ans=np.load("answer/answer.npy")
    sample=pd.read_csv("raw_data/sample.csv")
    sample.label=ans
    NUM=get_answer_num()
    print("the answer is ready at answer"+str(NUM)+".csv")
    sample.to_csv("answer/answer"+str(NUM)+".csv",index=False)