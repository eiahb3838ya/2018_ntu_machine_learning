# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:50:15 2018

@author: eiahb
"""

#import scipy,pprint
#from pprint import pprint
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.metrics import log_loss
#import datetime
from my_class.common_function import *
from imblearn.over_sampling import SMOTE, ADASYN,RandomOverSampler



TRAIN_NUM=10

mylog=init_logging()


w_init=np.load("temp_W_b/lr_W.npy")
b_init=np.load("temp_W_b/lr_b.npy")
#load prework of train_x
raw_train_x=pd.read_csv("train_x.csv",encoding="big5")
train_x=prework_x(raw_train_x)

#load prework of train_y
raw_train_y=pd.read_csv("train_y.csv",encoding="big5")
train_y=raw_train_y

#load prework of test_x
raw_test_x=pd.read_csv("test_x.csv",encoding="big5")
test_x=prework_x(raw_test_x)

#reshape to fit model
train_x_np=np.array(train_x)
train_y_np=np.array(train_y)#.reshape(-1,)
test_x_np=np.array(test_x)
print("shape of train_x,test_x,train_y_np:",train_x_np.shape,test_x_np.shape,train_y_np.shape)
#resampling
#x_resampled, y_resampled = SMOTE().fit_resample(train_x_np, train_y_np)
#print("shape of X_resampled,y_resampled:",x_resampled.shape,y_resampled.shape)

#train_x=x_resampled.reshape(-1,train_x_np.shape[1])
#train_y=y_resampled.reshape(-1,1)
#print("shape of train_x,train_y:",train_x.shape,train_y.shape)

lr=Logistic_Regression_gradient()
lr.train(train_x_np,train_y_np,train_num=TRAIN_NUM,w_init=w_init,b_init=b_init,epochs=5000000,batch_size=50)
mylog.info("training done")

test_x_scaled=lr.feature_scaling(test_x_np)
lr.predict(test_x_scaled,train_num=TRAIN_NUM,result=True)

np.save("temp_W_b/lr_W.npy",lr.W,)
np.save("temp_W_b/lr_b.npy",lr.b,)
#last_W=lr.W
#last_b=lr.b
#mylog.debug("start train #"+str(TRAIN_NUM))
#lr.train(train_x_np,train_y_np,w_init=last_W,b_init=last_b,train_num=TRAIN_NUM,epochs=500000)
#test_x=lr.feature_scaling(test_x)
#last_W=lr.W
#last_b=lr.b
#lr.predict(test_x,result=True,train_num=TRAIN_NUM)



#W = np.zeros((train_x.shape[1], 1))
#np.dot(train_x,W)
#sigmoid_v = np.vectorize(sigmoid)
#sigmoid_v(np.dot(train_x,W))
