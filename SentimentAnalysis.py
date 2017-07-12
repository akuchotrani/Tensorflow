# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:24:39 2017

@author: aakash.chotrani
"""

import nltk
from nltk.tokenize import word_tokenize
#i pulled the chai up to the table
#word_tokenize will seperate the workds
#[i, pulled , the, chair, up, to, the, table]
from nltk.stem import WordNetLemmatizer
# run, running runs 
import numpy as np 
import random
import pickle #save the data at some point
from collections import Counter#count some stuff

lemmatizer = WordNetLemmatizer()
HowMany_lines = 100000

def create_lexicon(pos,neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file,'r') as f:
            contents = f.readlines()
            for line in contents[:HowMany_lines]:
                all_words = word_tokenize(line.lower())
                lexicon += list(all_words)#this contains all the words and copies of all the words. Do we care about all the words?
                
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]#remove all the tense
    w_counts = Counter(lexicon) #Dictionary like elements how many times the words occur  {'the': 23656, 'and': 5445}
    
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:# we only care about the words which occur more frequently in certain range not too common not too scarce
           l2.append(w)
           
           
    return l2

def sample_handling(sample,lexicon,classification):
    feature_set = []
    with open(sample,'r') as f:
        contents = f.readlines()
        for line in contents[:HowMany_lines]:
            current_words = word_tokenize(line.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            #if the word in the line appears in our lexicon dictionary then increament the value by one
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            '''
                feature set will contain : [
                                            [1 0 0 1 1 0],
                                            [1 0] <-------------- positive sentiment else [0 1]<---------negative sentiment
                                           ]
            '''
            feature_set.append([features,classification])
    return feature_set


def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
    lexicon = create_lexicon(pos,neg)
    print(len(lexicon))
    features = []
    features += sample_handling('pos.txt',lexicon,[1,0])
    features += sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features)
    
    features = np.array(features)
    
    testing_size = int(test_size*len(features))
    
    
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x,train_y,test_x,test_y


if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)
    
    
    
    
    
    
            
            
                
                