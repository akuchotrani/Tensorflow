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

def create_lexicom(pos,neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file,'r') as f:
            contents = f.readlines()
            for line in contents[:HowMany_lines]:
                all_words = word_tokenize(line)
                lexicon += list(all_words)#this contains all the words and copies of all the words. Do we care about all the words?
                
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]#remove all the tense
    w_counts = Counter(lexicon) #Dictionary like elements how many times the words occur  {'the': 23656, 'and': 5445}
    
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:# we only care about the words which occur more frequently in certain range not too common not too scarce
           l2.append(w)
           
           
    return l2
            
                
                