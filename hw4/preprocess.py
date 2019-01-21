
# coding: utf-8

# In[427]:
from share_functions import *


import re,jieba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec,LineSentence
from tqdm import tqdm

# In[338]:






# In[462]:


PADDING_LENGTH=150
ROOT="D:/work/school/machine learning/hw/hw4/"

tqdm.pandas()
# # preprocess
# ## jieba

# In[85]:


file_name=ROOT+"data_set/dict.txt.big"
jieba.load_userdict(file_name)

# ## load and normalize

# In[449]:


X_train_raw=pd.read_csv(ROOT+"data_set/train_x.csv",encoding="utf8",sep='\n',header=0).loc[:,'id,comment']
X_test_raw=pd.read_csv(ROOT+"data_set/test_x.csv",encoding="utf8",sep='\n',header=0).loc[:,'id,comment']
y_train_raw=pd.read_csv(ROOT+"data_set/train_y.csv",encoding="utf8").set_index('id')
answer_sample=pd.read_csv(ROOT+"data_set/sample_submission.csv",encoding="utf8").set_index('id')

# In[418]:

print("start preprocess X_train")
X_train_split=get_X_split(X_train_raw)
X_train_resub=get_X_resub(X_train_split)
X_train_tokenize=get_X_tokenize(X_train_resub)
X_train_filter=get_X_filter(X_train_tokenize)

np.save(ROOT+"preprocessed/X_train_filter.npy",X_train_filter)


print("start preprocess X_test")
X_test_split=get_X_split(X_test_raw)
X_test_resub=get_X_resub(X_test_split)
X_test_tokenize=get_X_tokenize(X_test_resub)
X_test_filter=get_X_filter(X_test_tokenize)

np.save(ROOT+"preprocessed/X_test_filter.npy",X_test_filter)
# In[438]:
print("update w2v")
w2v_model=Word2Vec(pd.concat([X_train_tokenize,X_test_tokenize]),size=512,workers=12,sg=2)
# In[439]:

# pd.concat([X_train_tokenize,X_test_tokenize])
# X_train_tokenize[6]


# w2v_model.build_vocab(X_test_filter, update=True)
# w2v_model.train(X_test_filter,total_examples=w2v_model.corpus_count,epochs=200)


# In[440]:
w2v_model.save(ROOT+"preprocessed/w2v_model.model")
print("done yaho")



