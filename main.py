# -*- coding: utf8 -*-
import re
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn import linear_model

import preprocessing
import chunks

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', help='Path to vocabulary file', default="data/vocab.txt")#required=True)
parser.add_argument('--proc_train',type=bool, help='Should preprocess train', default=False)
parser.add_argument('--proc_test',type=bool, help='Should preprocess test', default=False)
parser.add_argument('--train', help='Path to training file', default="data/train.txt")#required=True)
parser.add_argument('--test', help='Path to test file', default="data/test.txt")#required=True)

args = parser.parse_args()
test_file, train_file = args.test, args.train

vocab = []
with open(args.vocab, "r") as f:
    s = f.read()
    vocab = s.split(',')

vectorizer = CountVectorizer(vocabulary=vocab)

def get_data(input, flag):
    data, labels = [], []
    if flag:
        train_data = preprocessing.process(test_file)
        data, labels = train_data['text'], train_data['labels']
    else:
        with open(train_file) as f:
            gen = chunks.read_chunk(f, "\n")
            # поправлю 40к когда придумаю на что
            for i in range(40000):
                s = next(gen).split('\t')
                data.append(s[-1])
                labels.append(s[-2])
    return data, labels
data, labels = get_data(train_file, args.proc_train)

features = vectorizer.fit_transform(data)
transformer = TfidfTransformer()
transformer.fit(features)
tfidf_features = transformer.transform(features)
clf = clf = linear_model.SGDClassifier(loss='squared_hinge')
clf = clf.fit(tfidf_features, labels)

# testing
test_data, test_labels = get_data(test_file, args.proc_test)

test_data_features = vectorizer.transform(test_data).toarray()
result = clf.predict(test_data_features)
check = zip(test_labels, result)
    
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