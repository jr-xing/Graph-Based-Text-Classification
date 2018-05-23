# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:48:46 2018

@author: remussn
"""

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

# Change the path according to your system
stanford_classifier = 'E:\\xdocuments\\Courses\\2018_Spring\\graph\\playground_yu\\masterProject\\graph\\stanford-ner-2018-02-27\\stanford-ner-2018-02-27\\classifiers\\english.all.3class.distsim.crf.ser.gz'
stanford_ner_path = 'E:\\xdocuments\\Courses\\2018_Spring\\graph\\playground_yu\\masterProject\\graph\\stanford-ner-2018-02-27\\stanford-ner-2018-02-27\\stanford-ner.jar'

# Creating Tagger Object
st = StanfordNERTagger(stanford_classifier, stanford_ner_path, encoding='utf-8')

text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'

tokenized_text = word_tokenize(text)
classified_text = st.tag(tokenized_text)

print classified_text

import os
java_path = "C:/Program Files/Java/jre1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
os.environ['JAVA_HOME'] = java_path

import nltk
nltk.internals.config_java("C:/Program Files/Java/jre1.8.0_161/bin/java.exe")

# 在 NLTK 中使用 Stanford NLP 工具包
# http://www.zmonster.me/2016/06/08/use-stanford-nlp-package-in-nltk.html

# Configuring Stanford Parser and Stanford NER Tagger with NLTK in python on Windows and Linux
# https://blog.manash.me/configuring-stanford-parser-and-stanford-ner-tagger-with-nltk-in-python-on-windows-f685483c374a

# Problem of NLTK with StanfordTokenizer
# https://tianyouhu.wordpress.com/2016/09/01/problem-of-nltk-with-stanfordtokenizer/

# representations:333
# nodeLabel: 251

## nodelabe 247: edit stadford path

# ASCII

# MATLAB, Java