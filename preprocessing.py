# -*- coding: utf8 -*-
import pymorphy2
import pandas as pd
import numpy as np
import re
import sys

stopwords = ["в", "и", "я", "а", "по", "с", "у", "на", "р", "ф", "", "тот", "за", "так", "же",
             "т", "к", "от", "при", "три", "пять"]

morph = pymorphy2.MorphAnalyzer()

def f(word):
    return morph.parse(word)[0].normal_form

def process_table(table):
    data = table[1].str.split(',', 2).apply(pd.Series, 1)
    data = data[data[2] != ""]
    data[2] = data[2].apply(lambda x: str(x)[9:-3])
    data.columns = ["id", "label", "text"]
    data['label'] = data['label'].apply(lambda x: '1' if x == '3' or x == '4' else '0')
    return data

def process_text(text):
    sample = re.split('\W+', re.sub('\W(Н|н)(е|Е) ', ' не', text))
    l = [f(x) for x in sample if f(x) not in stopwords and not re.match("[0-9]+", x)]
    text = ' '.join(l)
    return text

def process(path):
    data = pd.read_csv(path, delimiter='autoru-', header = None, quoting=3, engine='python')
    data = process_table(data)
    data['text'] = data['text'].apply(lambda x: process_text(x))
    return data
