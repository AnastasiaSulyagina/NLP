import pandas as pd
import numpy as np
import sys
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

import preprocessing


train_file, test_file = sys.argv[1], sys.argv[2]

data = preprocessing.process(train_file)

vectorizer = TfidfVectorizer(analyzer='word', min_df = 0)
train_data_features =  vectorizer.fit_transform(data['text'])

def show_word_scores():
    words = vectorizer.get_feature_names()
    dense = train_data_features.todense()
    scores = dense[0].tolist()[0]
    phrase_scores = sorted([(vocab[num], score) for (num, score) in zip(range(0, len(scores)), scores) if score > 0],
                           key=lambda t: t[1] * -1)
    for word, score in phrase_scores:
        print (score, word)
#show_word_scores()

clf = linear_model.LinearRegression()
clf = clf.fit( train_data_features, data['label'] )

#testing
test_data = preprocessing.process(test_file)
test_data_features = vectorizer.transform(test_data['text'])

result = clf.predict(test_data_features)

check = zip(test_data['label'], result)
tp, tn, fp, fn = 0, 0, 0, 0

def compare_results():
    print("value, prediction:")
    for value, prediction in check:
        print (value, prediction)
    
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
