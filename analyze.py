# -*- coding: utf8 -*-
import re
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import linear_model
from scipy import interp

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
test_data, test_labels = get_data(test_file, args.proc_test)
data = data + test_data
labels = labels + test_labels
features = vectorizer.fit_transform(data)
test_data_features = vectorizer.transform(test_data).toarray()

transformer = TfidfTransformer()
tfidf_features = transformer.fit(features).transform(features)

# testing
def show_roc():

    X = np.array(tfidf_features.todense())
    y = np.array(labels).astype(int)
    cv = StratifiedKFold(y, n_folds=5)
    classifier = linear_model.SGDClassifier(loss='squared_hinge')
    for i, (train, test) in enumerate(cv):
        res = classifier.fit(X[train], y[train]).predict(X[test])

        fpr, tpr, _ = roc_curve(y[test], res)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        check = zip(y[test], res)
        tp, tn, fp, fn = 0, 0, 0, 0
        for value, prediction in check:
            if value == prediction:
                tp += value # value 1 or 0
                tn += 1 - value
            else:
                fn += value
                tn += 1 - value
        print ('TP: {0}, TN: {1}, FP: {2}, FN: {3}'.format(tp, tn, fp, fn))
        print ("precision: " + str(tp / (tp + fp if tp + fp != 0 else 0.0000000000001)))
        print ("recall: " + str(tp / (tp + fn if tp + fn != 0 else 0.0000000000001)))


    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
show_roc()