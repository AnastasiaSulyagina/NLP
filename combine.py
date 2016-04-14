# -*- coding: utf8 -*-
import re
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn import linear_model, metrics
from sklearn.pipeline import make_pipeline
from sklearn.svm import *
from scipy import interp
from math import log

#import preprocessing
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

stopwords_file = "data/stopwords.txt"
stopwords = []
with open(stopwords_file, "r") as f:
    s = f.read()
    stopwords = s.split(',')

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
test_data, test_labels = get_data(test_file, args.proc_test)
data, labels  = np.array(data), np.array(labels).astype(int)
inv_labels = np.logical_not(labels)
test_data, test_labels  = np.array(test_data), np.array(test_labels).astype(int)

def get_word_frequencies(some_data, out_file):
    word_freq = {}
    vectorizer = CountVectorizer(vocabulary=vocab)
    data_features = vectorizer.fit_transform(some_data).toarray()
    words = vectorizer.get_feature_names()
    frequencies = np.sum(data_features, axis=0)
    #with open(out_file, "w+") as f:
    for i, (fr, word) in enumerate(sorted(zip(frequencies, words), reverse=True)):
        if word not in stopwords:
            #f.write(str(fr) + ' ' + word + '\n')
            word_freq[word] = i + 1
    return word_freq

good = get_word_frequencies(data[labels], "data/clean_good_vocab.txt")
bad = get_word_frequencies(data[inv_labels], "data/clean_bad_vocab.txt")

def preprocess(data):
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] != 0:
                val = (bad[vocab[j]] if vocab[j] in bad else len(bad) / (good[vocab[j]] if vocab[j] in good else len(good)))
                data[i][j] = log(data[i][j] * val)
            if not np.isfinite(data[i][j]):
                print(vocab[j])
    return data

def boost(X, y, X_test, y_test):
    clf1 = RandomForestClassifier(n_estimators=10)

    clf2 = XGBClassifier(learning_rate =0.05, n_estimators=150, max_depth=6,
     min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,
     objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
    pipeline = make_pipeline(clf1, clf2)
    pipeline.fit(X, y)

    prob = pipeline.predict_proba(X_test)[:, 1]
    res = pipeline.predict(X_test)
    print ("Precision Score : %f" % metrics.precision_score(y_test, res))
    print ("Recall Score : %f" % metrics.recall_score(y_test, res))
    return roc_curve(y_test, prob)

def stack(X, y, X_test, y_test):
    X, X1, y, y1 = train_test_split(X, y, test_size=0.5)
    #clf1 = GradientBoostingClassifier(n_estimators=10)
    clf1 = RandomForestClassifier(n_estimators=20)
    enc = OneHotEncoder()
    clf2 = RandomForestClassifier(n_estimators=10)
    clf1.fit(X, y)
    enc.fit(clf1.apply(X))
    clf2.fit(enc.transform(clf1.apply(X1)), y1)

    #prob = clf2.predict_proba(enc.transform(clf1.apply(X_test)[:, :, 0]))[:, 1]
    prob = clf2.predict_proba(enc.transform(clf1.apply(X_test)))[:, 1]
    res = clf2.predict(enc.transform(clf1.apply(X_test)))
    print ("Precision Score : %f" % metrics.precision_score(y_test, res))
    print ("Recall Score : %f" % metrics.recall_score(y_test, res))
    return roc_curve(y_test, prob)

vectorizer = CountVectorizer(vocabulary=vocab)
features = vectorizer.fit_transform(data)
#transformer = TfidfTransformer()
#tfidf_features = transformer.fit(features).transform(features)

t_features = vectorizer.transform(test_data)
X = features.toarray()

fpr, tpr, _ = stack(X, labels, t_features, test_labels)

def draw():
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
draw()
