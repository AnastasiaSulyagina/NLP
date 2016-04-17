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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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

def get_data(input_file, flag):
    data, labels = [], []
    if flag:
        all_data = preprocessing.process(input_file)
        data, labels = np.array(all_data['text']), np.array(all_data['labels']).astype(int)
    else:
        with open(input_file) as f:
            gen = chunks.read_chunk(f, "\n")
            for i in range(40000):
                s = next(gen).split('\t')
                data.append(s[-1])
                labels.append(s[-2])
    return np.array(data), np.array(labels).astype(int)

data, labels = get_data(train_file, args.proc_train)
test_data, test_labels = get_data(test_file, args.proc_test)
inv_labels = np.logical_not(labels)

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
    #clf1 = RandomForestClassifier(n_estimators=20)
    clf1 = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
    clf2 = linear_model.SGDClassifier(loss='log')
    enc = OneHotEncoder()
    #clf2 = RandomForestClassifier(n_estimators=10)
    #clf2 = GradientBoostingClassifier(n_estimators=20)
    clf1.fit(X, y)
    enc.fit(clf1.apply(X))
    clf2.fit(enc.transform(clf1.apply(X1)), y1)

    #prob = clf2.predict_proba(enc.transform(clf1.apply(X_test)[:, :, 0]))[:, 1]

    prob = clf2.predict_proba(enc.transform(clf1.apply(X_test)).toarray())[:, 1]
    res = clf2.predict(enc.transform(clf1.apply(X_test)))        
    check = zip(y_test, res)
    tp, tn, fp, fn = 0, 0, 0, 0
    for value, prediction in check:
        if (prediction and value):
            tp += 1
        if (prediction and not value):
            fp += 1
        if (not prediction and value):
            fn += 1
        if (not prediction and not value):
            tn += 1
    print ('TP: {0}, TN: {1}, FP: {2}, FN: {3}'.format(tp, tn, fp, fn))
    print ("Precision Score : %f" % metrics.precision_score(y_test, res))
    print ("Recall Score : %f" % metrics.recall_score(y_test, res))
    return roc_curve(y_test, prob)
def vote(X, y, X_test, y_test):
    X, X1, y, y1 = train_test_split(X, y, test_size=0.5)
    clf1 = linear_model.SGDClassifier(loss='squared_hinge')
    clf2 = XGBClassifier(learning_rate =0.03, n_estimators=150, max_depth=6,
                         min_child_weight=4, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
    clf3 = GaussianNB()
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
        clf.fit(X, y)
        res = clf.predict(X_test)
        print ("Precision Score : %f" % metrics.precision_score(y_test, res))
        print ("Recall Score : %f" % metrics.recall_score(y_test, res))

vectorizer = CountVectorizer(vocabulary=vocab)
features = vectorizer.fit_transform(data)
transformer = TfidfTransformer()
tfidf_features = transformer.fit(features).transform(features)

t_features = vectorizer.transform(test_data)
X = tfidf_features.toarray()

#fpr, tpr, _ = 
vote(X, labels, t_features.toarray(), test_labels)

def draw():
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='plot')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
draw()
