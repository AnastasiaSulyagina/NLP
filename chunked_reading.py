# -*- coding: utf8 -*-
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
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
        chunk = f.read(1024)
        if not chunk:
            yield buf
            break
        buf += chunk

l = []
with open(data_path + 'processed_train.txt') as f:
    gen = myreadlines(f, "\n")
    for i in range(5):
        l.append(next(gen))
    print(l)