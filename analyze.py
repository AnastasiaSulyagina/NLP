# -*- coding: utf8 -*-
import re
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn import linear_model, metrics
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
data, labels  = np.array(data + test_data), np.array(labels + test_labels).astype(int)
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

# testing
def show_roc(classifier, with_probas):
    cv = StratifiedKFold(labels[:-1], n_folds=5)

    for i, (train, test) in enumerate(cv):

        vectorizer = CountVectorizer(vocabulary=vocab)
        features = vectorizer.fit_transform(data[train])
        #transformer = TfidfTransformer()
        #tfidf_features = transformer.fit(features).transform(features)
        #X = np.array(tfidf_features.todense())

        #X = preprocess(features.toarray())
        X = features.toarray()
        y = labels[train]

        X, X1, y, y1 = train_test_split(X, y, test_size=0.5)
        clf1 = RandomForestClassifier(n_estimators=20)
        enc = OneHotEncoder()
        clf2 = RandomForestClassifier(n_estimators=10)
        clf1.fit(X, y)
        enc.fit(clf1.apply(X))
        clf2.fit(enc.transform(clf1.apply(X1)), y1)

      
        #clf = classifier.fit(X, y)

        X_test = vectorizer.transform(data[test])
        #t_f = preprocess(t_features.toarray())
        y_test = labels[test]
        #res = clf.predict(t_f)
        
        res = clf2.predict(enc.transform(clf1.apply(X_test)))  

        if with_probas:
            res_p = clf2.predict_proba(enc.transform(clf1.apply(X_test)))
            #res_p = clf.predict_proba(t_features)
            fpr, tpr, _ = roc_curve(y_test, res_p[:,1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

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
        print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, res))
        print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, res))

    if with_probas:
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

#show_roc(XGBClassifier(learning_rate =0.03, n_estimators=150, max_depth=6,
# min_child_weight=4, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), True)

#show_roc(linear_model.SGDClassifier(loss='squared_hinge'), False)
#show_roc(RandomForestClassifier(n_estimators=10), True)
show_roc(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), True)
#show_roc(ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0), True)