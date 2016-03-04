#!/usr/bin/python
# -*- coding: utf-8 -*-
# Trains nltk tokenizer using N random russian wiki texts

import codecs
import pickle
import urllib.request
import nltk.data
from bs4 import BeautifulSoup
from nltk.tokenize.punkt import PunktSentenceTokenizer


def main():
    collect_wiki_corpus('russian', 'ru', 1000)
    train_sentence_splitter('russian')

def get_random_article(namespace=None):
    """ Download a random wikipiedia article"""
    try:
        url = 'http://ru.wikipedia.org/wiki/Special:Random'
        if namespace != None:
            url += '/' + namespace
        req = urllib.request.Request(url, None, { 'User-Agent' : 'x'})
        page = urllib.request.urlopen(req).read()
        return page
    except (urllib2.HTTPError, urllib2.URLError):
        print ("Failed to get article")
        raise

def collect_wiki_corpus(language, lang, num_items):
    """
    Download <n> random wikipedia articles in language <lang>
    """
    filename = "%s.plain" % (language)
    out = codecs.open(filename, "w", "utf-8")

    for i in range(num_items):
        article_dict = get_random_article()
        
        # Remove html, styling and formatting
        soup = BeautifulSoup(article_dict)
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.body.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
    
        p_text = ''
        for p in soup.findAll('p'):
            only_p = p.findAll(text=True)
            p_text = ''.join(only_p)

            # Tokenize but keep . at the end of words
            p_tokenized = ' '.join(PunktSentenceTokenizer().tokenize(p_text))

            out.write(p_tokenized)
            out.write("\n")

    out.close()


def train_sentence_splitter(lang):
    """
    Train an NLTK punkt tokenizer for sentence splitting.
    http://www.nltk.org
    """
    # Read in trainings corpus
    plain_file = "%s.plain" % (lang)
    text = codecs.open(plain_file, "Ur", "utf-8").read()

    # Train tokenizer
    tokenizer = PunktSentenceTokenizer()
    tokenizer.train(text)

    # Dump pickled tokenizer
    pickle_file = "%s.pickle" % (lang)
    out = open(pickle_file, "wb")
    pickle.dump(tokenizer, out)
    out.close()


if __name__ == "__main__":
    main()
