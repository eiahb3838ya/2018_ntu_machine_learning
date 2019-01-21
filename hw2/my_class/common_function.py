# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:37:52 2018

@author: eiahb
"""

from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging,math,datetime



def init_logging():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        handlers = [logging.FileHandler('training_log/train.log', 'a', 'utf-8'),])
    # 定義 handler 輸出 sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 設定輸出格式
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # handler 設定輸出格式
    console.setFormatter(formatter)
    # 加入 hander 到 root logger
    logging.getLogger('').addHandler(console)
    return(logging)

def add_one_hot(x,target_column_name,combine_colunm_num=None):
    dummy_columns=pd.get_dummies(x[target_column_name],prefix=target_column_name)
    combine_target=dummy_columns.iloc[:,combine_colunm_num:]
    dummy_columns=dummy_columns.drop(combine_target,axis=1)
    other_column=combine_target.sum(axis=1)
    other_column.name=target_column_name+"_other"
    dummy_columns=pd.concat([dummy_columns,other_column],axis=1)
    out=x.drop(target_column_name,axis=1)
    out=pd.concat([out,dummy_columns],axis=1)
    return(out)
def prework_x(raw_x):
    new_x=raw_x
    new_x['MARRIAGE']=new_x['MARRIAGE'].replace(2,0)
    new_x['SEX']=new_x['SEX'].replace(2,0)
    new_x=new_x.drop(["MARRIAGE","SEX","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"],axis=1)
    turn_dummy=['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    for dum in turn_dummy:
        new_x=add_one_hot(new_x,dum,6)
    return(new_x)

def sigmoid(x):
    return (1 / (1 + math.exp(-x)))

class Logistic_Regression_gradient():
    def __init__(self):
        pass
    def parameter_init(self, dim,w_init=None,b_init=None):
        if  b_init:
            self.W = w_init
            self.b = b_init
        else:
            self.W = np.zeros((dim, 1))
#            print(self.W)
            self.b = 0
    def feature_scaling(self, X, train=False):    
        if train:
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
        return (X - self.min) / (self.max - self.min)
    
    def z_feature_scaling(self,X,train=False):
        if train:
            self.mean=np.mean(X,axis=0)
            self.std=np.std(X,axis=0)
        return((X-self.mean)/self.std)
        
    def predict(self, X,result=False,train_num=-1): 
        z=np.dot(X, self.W) + self.b
        # define vectorized sigmoid
#        print("z",z)
        sigmoid_v = np.vectorize(sigmoid)

#         # test
#         scores = np.array([ -0.54761371,  17.04850603,   4.86054302])
#         print sigmoid_v(scores)
        out=sigmoid_v(z)
        if result:
            answer=np.round(out).reshape(1,-1)[0]
            id_index=pd.Index(["id_"+str(i) for i in range(len(X))])
            out_df=pd.DataFrame({"Value":answer.astype(int)},index=id_index)
            datetime_str=str(datetime.date.today())
            if train_num>=0:
                out_df.to_csv("result_"+str(train_num)+"_"+datetime_str+".csv",index_label="id")
            else:
                out_df.to_csv("result"+datetime_str+".csv",index_label="id")
#        print("sigmoid z",out)
        return (out)
        
    def RMSELoss(self, X, Y):
        return np.sqrt(np.mean((Y - self.predict(X))** 2) )
    
    def cross_entropy(self,X,Y):
        pred=self.predict(X)
#         pred=[[i,1-i] for i in pred]
        return(log_loss(Y,pred))
    
    def train(self, X, Y,train_num, epochs=30000, lr=0.01 ,batch_size=None ,w_init=None,b_init=None,feature_scaling="feature_scaling",info=None,): 
        logging.info("start train num :"+str(train_num)+" feature_scaling:"+feature_scaling+"  epoch:"+str(epochs))
        sample_size=X.shape[0]
        if not batch_size:
            batch_size = sample_size
        
        W_dim = X.shape[1]
        
        self.parameter_init(W_dim,w_init=w_init,b_init=b_init)
        
        if feature_scaling=="z_feature_scaling":
            X = self.z_feature_scaling(X, train=True)
        else:
            X = self.feature_scaling(X, train=True)

            
        lr_b = 0.00000001
        lr_W = np.zeros((W_dim, 1))+0.00000000001

        loss_list=list()
        for epoch in range(epochs):
            batch_rand_int=np.random.randint(low=0,high=sample_size,size=batch_size)
#            print(batch_rand_int)
            batch_X=X[batch_rand_int]
            batch_Y=Y[batch_rand_int]
            if not epoch%5000:
                loss=self.cross_entropy(X,Y)
                loss_list.append(loss)
                print(epoch,loss)
#                plt.plot(loss_list[-10:-1])
#                plt.show()
                record_attr=np.insert(self.W,0,self.b)
                record_attr=np.insert(record_attr,0,loss)
                record_attr=record_attr.reshape(1,-1)
                pd.DataFrame(record_attr).to_csv("training_log/train"+str(train_num)+".csv",mode='a',index=False,header=False)
            
            # mse loss
            grad_b = -np.sum(batch_Y - self.predict(batch_X))/ batch_size
            
#            print("grad_b",np.sum(batch_Y - self.predict(batch_X)))
            grad_W = -np.dot(batch_X.T, (batch_Y - self.predict(batch_X))) / batch_size
#            print("grad_W",-np.dot(batch_X.T, (batch_Y - self.predict(batch_X))) / batch_size)
#            print(self.predict(X),batch_Y)
            # adagrad
            lr_b += grad_b ** 2
            lr_W += grad_W ** 2
#            print("lr_b5454545",lr_b)
            #update
            self.b = self.b - lr / np.sqrt(lr_b) * grad_b
            self.W = self.W - lr / np.sqrt(lr_W) * grad_W
#            print("11111111",self.b)
#