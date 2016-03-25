# -*- coding: utf8 -*-
import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model

import preprocessing


train_file, test_file = sys.argv[1], sys.argv[2]

data = preprocessing.process(train_file)

vectorizer = CountVectorizer(analyzer = "word", max_features = 2000)
train_data_features = vectorizer.fit_transform(data['text']).toarray()

def show_word_frequencies(out_file, print_data):
    vectorizer = CountVectorizer(analyzer = "word", max_features = 2000)
    data_features = vectorizer.fit_transform(print_data['text']).toarray()
    words = vectorizer.get_feature_names()
    frequencies = np.sum(data_features, axis=0)
    with open(out_file, "w+") as f:
        for fr, word in sorted(zip(frequencies, words), reverse=True):
            f.write(str(fr) + word + '\n')

data[data['label'] == '1'].to_csv('bad_vocab.txt', sep='\t', encoding='utf-8')
data[data['label'] == '0'].to_csv('good_vocab.txt', sep='\t', encoding='utf-8')
show_word_frequencies("bad_features.txt", data[data['label'] == '1'])
show_word_frequencies("good_features.txt", data[data['label'] == '0'])

clf = linear_model.LinearRegression()
clf = clf.fit( train_data_features, data['label'] )

## testing
test_data = preprocessing.process(test_file)

test_data_features = vectorizer.transform(test_data['text']).toarray()
result = clf.predict(test_data_features)
check = zip(test_data['label'], result)

def compare_results():
    print("value, prediction:")
    for value, prediction in check:
        print (value, prediction)
    
tp, tn, fp, fn = 0, 0, 0, 0

for value, prediction in check:
    if value == prediction:
        if value == '1':
            tp += 1
        else:
            tn += 1
    else:
        if value == '1':
            fn += 1
        else:
            tn += 1

print('TP: {0}, TN: {1}, FP: {2}, FN: {3}'.format(tp, tn, fp, fn))
#print ("precision: " + str(tp / (tp + fp)))
#print ("recall: " + str(tp / (tp + fn)))