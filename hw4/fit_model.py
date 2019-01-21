from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense ,LSTM,BatchNormalization,Dropout,LeakyReLU,Bidirectional,CuDNNLSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras import backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec

from share_functions import *

PADDING_LENGTH=150
ROOT="D:/work/school/machine learning/hw/hw4/"
EPOCHS=1000
BATCH_SIZE=100

# import preprocess

w2v_model=Word2Vec.load(ROOT+"preprocessed/w2v_model.model")
embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
word2idx = {}
vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1


# ## test_to_index、padding 
# ### prepare X_train、y_train

# In[326]:
X_train_filter=np.load(ROOT+"preprocessed/X_train_filter.npy")

X_train_index=get_X_index(X_train_filter,word2idx)
X_train_padding=get_X_padding(X_train_index)
X_train=X_train_padding
print("X_train_Shape:", X_train.shape)


# In[327]:
y_train_raw=pd.read_csv(ROOT+"data_set/train_y.csv",encoding="utf8").set_index('id')
y_train=np.array(y_train_raw).reshape(-1,1)
print("y_Shape:", y_train.shape)


# ### prepare X_test

# In[441]:
X_test_filter=np.load(ROOT+"preprocessed/X_test_filter.npy")

X_test_index=get_X_index(X_test_filter,word2idx)
X_test_padding=get_X_padding(X_test_index)
X_test=X_test_padding
print("X_test_Shape:", X_test.shape)


# ## use embedding_matrix above build embedding layer and model

# In[311]:


embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False)


# In[342]:
backend.clear_session()

model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True,kernel_initializer='orthogonal', recurrent_initializer='orthogonal')))
model.add(Dropout(0.5))
model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True,kernel_initializer='orthogonal', recurrent_initializer='orthogonal')))
model.add(Dropout(0.6))
model.add(Bidirectional(CuDNNLSTM(128,kernel_initializer='orthogonal', recurrent_initializer='orthogonal')))
model.add(Dropout(0.6))
# model.add(LSTM(128))

# model.add(Dense(512))
# model.add(LeakyReLU(0.3))
# model.add(Dropout(0.3))

# model.add(Dense(256))
# model.add(LeakyReLU(0.3))
# model.add(Dropout(0.5))

model.add(Dense(128))
model.add(LeakyReLU(0.3))
model.add(BatchNormalization())
model.add(Dropout(0.6))

model.add(Dense(128))
model.add(LeakyReLU(0.3))
model.add(BatchNormalization())
model.add(Dropout(0.6))

model.add(Dense(64))
model.add(LeakyReLU(0.3))
model.add(BatchNormalization())
model.add(Dropout(.65))#3

model.add(Dense(1, activation='sigmoid'))


early_stopping=EarlyStopping(monitor='val_loss', min_delta=0,
                              patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.35,
                              patience=2, min_lr=0.00001,verbose=1,)


adam=Adam(lr=0.0001)

model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'],
              )
print(model.summary())


# In[343]:


history = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1,callbacks=[early_stopping,reduce_lr],verbose = 1)

# In[445]:


answer_np=model.predict_classes(X_test)
np.save(ROOT+"answer/answer.npy",answer_np)
answer_to_csv()