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
from xgboost.sklearn import XGBClassifier
from sklearn import linear_model, metrics
from sklearn.svm import *
from scipy import interp

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
#test_data, test_labels = get_data(test_file, args.proc_test)
data, labels  = np.array(data), np.array(labels).astype(int)
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
                data[i][j] = np.float32(data[i][j] * val)
            if not np.isfinite(data[i][j]):
                print(vocab[j])

# testing
def show_roc(classifier, with_probas):
    cv = StratifiedKFold(labels, n_folds=5)

    for i, (train, test) in enumerate(cv):
        vectorizer = CountVectorizer(vocabulary=vocab)
        features = vectorizer.fit_transform(data[train])
        #transformer = TfidfTransformer()
        #tfidf_features = transformer.fit(features).transform(features)
        #X = np.array(tfidf_features.todense())

        X = preprocess(features.toarray())
        y = labels[train]
        # падает тут, проблема где-то в preprocess
        clf = classifier.fit(X, y)

        t_features = vectorizer.transform(data[test])
        t_f = preprocess(t_features.toarray())
        t_labels = labels[test]
        res = clf.predict(t_features)

        if with_probas:
            res_p = clf.predict_proba(t_features)
            fpr, tpr, _ = roc_curve(t_labels, res_p[:,1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        check = zip(y[test], res)
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
        print ("Precision Score : %f" % metrics.precision_score(t_labels, res))
        print ("Recall Score : %f" % metrics.recall_score(t_labels, res))

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

show_roc(linear_model.SGDClassifier(loss='squared_hinge'), False)