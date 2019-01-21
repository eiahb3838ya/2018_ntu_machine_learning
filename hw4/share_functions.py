import re,jieba
from keras.preprocessing.sequence import pad_sequences
from progressbar import progressbar
import numpy as np
import pandas as pd
PADDING_LENGTH=800
ROOT="D:/work/school/machine learning/hw/hw4/"

#jieba
file_name=ROOT+"data_set/dict.txt.big"
jieba.load_userdict(file_name)
jieba.add_word("^^")
jieba.add_word("XD")
jieba.add_word("= =")
jieba.add_word("累了")
jieba.add_word("頑game")
jieba.add_word(">////<",1000)
jieba.add_word("ㄏㄏ")
jieba.add_word("原po")
jieba.add_word("(¬_¬)ﾉ'")
jieba.add_word("")
jieba.add_word("")
jieba.add_word("")

#stopwords
stopwords=list()
with open(ROOT+'data_set/stop_word_punctuation.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stopwords.append(data)

#def functions
def get_X_split(X_raw):
    X_split=X_raw.progress_apply(lambda row:",".join(row.split(',')[1:]))
    return(X_split)

def get_X_resub(X_split):
    X_resub=X_split.progress_apply(lambda row:re.sub("[Bb][0-9]+","myAtSomebodySpecial,",row))
    X_resub=X_resub.progress_apply(lambda row:re.sub("[0-9]+","myNumberSpecial,",row))
    X_resub=X_resub.progress_apply(lambda row:re.sub("[.{2,}]+","......",row ,count=1))
    X_resub=X_resub.progress_apply(lambda row:re.sub("[\s]+","",row))
    return(X_resub)

def get_X_tokenize(X_resub):
    print("start get_X_tokenize")

    X_tokenize=X_resub.apply(jieba.lcut)
    return(X_tokenize)

def get_X_filter(X_tokenize):
    print("start get_X_filter")
    X_filter=X_tokenize.apply(lambda row:list(filter(lambda word :word not in stopwords,row)))
    return(X_filter)

def get_X_padding(X_index):
    X_padding=pad_sequences(X_index,PADDING_LENGTH,padding='pre')
    return(X_padding)

def get_answer_num():
    with open(ROOT+"answer/answer_number.txt","r") as p:
        NUM=p.read()
        
    NUM=int(NUM)+1
    with open(ROOT+"answer/answer_number.txt","w")as p:
        p.write(str(NUM))
      
    return(NUM)

def answer_to_csv(): 
    ans=np.load(ROOT+"answer/answer.npy")
    sample=pd.read_csv(ROOT+"data_set/sample_submission.csv").set_index('id')
    sample.label=ans
    NUM=get_answer_num()
    print("the answer is ready at answer"+str(NUM)+".csv")
    sample.to_csv(ROOT+"answer/answer_"+str(NUM)+".csv")

def get_X_index(corpus,word2idx):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)