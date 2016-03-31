# -*- coding: utf8 -*-
import re
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

import preprocessing

data_path = 'data/'

vocab = []
s = ''
with open(data_path + "vocab.txt", "r") as f:
    s = f.read()
    vocab = s.split(',')

vectorizer = CountVectorizer(vocabulary=vocab)

def myreadlines(f, newline):
    buf = ""
    while True:
        while newline in buf:
            pos = buf.index(newline)
            yield buf[:pos]
            buf = buf[pos + len(newline):]
        chunk = f.read(2048)
        if not chunk:
            yield buf
            break
        buf += chunk

data = []
labels = []
with open(data_path + 'processed_train.txt') as f:
    gen = myreadlines(f, "\n")
    for i in range(40000):
        s = next(gen).split('\t')
        data.append(s[-1])
        labels.append(s[-2])

features = vectorizer.fit_transform(data)
transformer = TfidfTransformer(norm="l2")
transformer.fit(features)
tfidf_features = transformer.transform(features)
clf = LinearSVC()
clf = clf.fit(tfidf_features, labels)

# testing
test_data = []
test_labels = []
with open(data_path + 'processed_test.txt') as f:
    gen = myreadlines(f, "\n")
    for i in range(40000):
        s = next(gen).split('\t')
        test_data.append(s[-1])
        test_labels.append(s[-2])

test_data_features = vectorizer.transform(test_data).toarray()
result = clf.predict(test_data_features)
check = zip(test_labels, result)

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
print ("precision: " + str(tp / (tp + fp)))
print ("recall: " + str(tp / (tp + fn)))