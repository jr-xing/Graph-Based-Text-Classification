# -*- coding: utf-8 -*-
"""
# https://stackoverflow.com/questions/8476805/recommendations-for-using-graphs-theory-in-machine-learning?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
architecture:
1. tograph(singletext)
2. kernerl(text1, text2)
    # it's not reasonable to read all - including test - texts. Maybe save a personal dictionary and filter out uncommon words?
    fit(corpus)     # build dictionary(tokenList)

"""
import re
def getNewsGroupContent(scriptList):
    stopBeginners = ['From','Subject','Nntp-Posting-Host','Organization',
                     'X-Newsreader','Lines']
#    scriptList = ['From: bil@okcforum.osrhe.edu (Bill Conner)\nSubject: Re: Not the Omni!\nNntp-Posting-Host: okcforum.osrhe.edu\nOrganization: Okcforum Unix Users Group\nX-Newsreader: TIN [version 1.1 PL6]\nLines: 18\n\nCharley Wingate (mangoe@cs.umd.edu) wrote:\n: \n: >> Please enlighten me.  How is omnipotence contradictory?\n: \n: >By definition, all that can occur in the universe is governed by the rules\n: >of nature. Thus god cannot break them. Anything that god does must be allowed\n: >in the rules somewhere. Therefore, omnipotence CANNOT exist! It contradicts\n: >the rules of nature.\n: \n: Obviously, an omnipotent god can change the rules.\n\nWhen you say, "By definition", what exactly is being defined;\ncertainly not omnipotence. You seem to be saying that the "rules of\nnature" are pre-existant somehow, that they not only define nature but\nactually cause it. If that\'s what you mean I\'d like to hear your\nfurther thoughts on the question.\n\nBill\n',
#                  "From: jhwitten@cs.ruu.nl (Jurriaan Wittenberg)\nSubject: Re: Magellan Update - 04/16/93\nOrganization: Utrecht University, Dept. of Computer Science\nKeywords: Magellan, JPL\nLines: 29\n\nIn <19APR199320262420@kelvin.jpl.nasa.gov> baalke@kelvin.jpl.nasa.gov \n(Ron Baalke) writes:\n\n>Forwarded from Doug Griffith, Magellan Project Manager\n>\n>                        MAGELLAN STATUS REPORT\n>                            April 16, 1993\n>\n>\n>2.  Magellan has completed 7225 orbits of Venus and is now 39 days from\n>the end of Cycle-4 and the start of the Transition Experiment.\nSorry I think I missed a bit of info on this Transition Experiment. What is it?\n\n>4.  On Monday morning, April 19, the moon will occult Venus and\n>interrupt the tracking of Magellan for about 68 minutes.\nWill this mean a loss of data or will the Magellan transmit data later on ??\n\nBTW: When will NASA cut off the connection with Magellan?? Not that I am\nlooking forward to that day but I am just curious. I believe it had something\nto do with the funding from the goverment (or rather _NO_ funding :-)\n\nok that's it for now. See you guys around,\nJurriaan.\n \n-- \n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n|----=|=-<- - - - - - JHWITTEN@CS.RUU.NL- - - - - - - - - - - - ->-=|=----|\n|----=|=-<-Jurriaan Wittenberg- - -Department of ComputerScience->-=|=----|\n|____/|\\_________Utrecht_________________The Netherlands___________/|\\____|\n"]
    scriptListProced = []
    for script in scriptList:        
        content = ''
        for line in script.split('\n'):
            if (len(line)>1)&(not line.startswith(tuple(stopBeginners))):
                content = content + re.sub(r'[^\w\s]','',line) + '\n'
        scriptListProced.append(content)
    return scriptListProced

#%% 1. Read data
import pandas as pd
# corpus_name = 'demo_mini'
corpus_name = 'state_of_the_union_corpus'
# corpus_name = 'simpsons_proc_3'
# corpus_name = '20newsgroups'

if corpus_name == 'demo_mini':
    data_raw = {
            'texts':
                ['one two one two three four one two', 'cat dog cat dog panda lion cat dog'],
            'labels':
                ['digit', 'animal']
            }
    texts_header = 'texts'
    labels_header = 'labels'
elif (corpus_name == 'state_of_the_union_corpus')|((corpus_name == 'state_of_the_union_corpus_para')):
    data_raw = pd.read_csv('./data/'+corpus_name+'.csv')
    texts_header = 'script'
    labels_header = 'president'
elif corpus_name == 'simpsons_proc_3':
    data_raw = pd.read_csv('./data/simpsons_proc_3.csv')
    texts_header = 'normalized_text'
    labels_header = 'raw_character_text'    
elif corpus_name == '20newsgroups':
    from sklearn.datasets import fetch_20newsgroups
    # from main_tuning_util import getNewsGroupContent
    cats = ['alt.atheism', 'sci.space']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
    texts_header = 'scripts'
    labels_header = 'labels'
    labels = [newsgroups_train.target_names[i] for i in newsgroups_train.target]
    data_raw = {texts_header:getNewsGroupContent(newsgroups_train.data),labels_header:labels}


# data_raw = data_raw.sample(frac = 0.2)

#%% 2. Extract texts and labels
texts = list(data_raw[texts_header])
labels = list(data_raw[labels_header])

#%% 2.1 Remove unfrequent words


#%% 3. Initialize graph

# **********  Fixed  **********
punc_tf = True
stpw_tf = False
vec_win_size = 10
vec_model = 0

# ********** Hyperparameters **********
graph_win_size = 3
vec_dim = 5
node_attr_type = 'word2vec' # word2vec
bow_model = 'tfidf'
kernel_type = 'rbf' # rbf

from modules.representations import textGraph
graph_undir = textGraph(punc = punc_tf, stpw = stpw_tf,
                  winSize = graph_win_size,
                  nd_attr_type = node_attr_type,
                  vec_dim = vec_dim, 
                  vec_win_size = vec_win_size, 
                  vec_model = vec_model, 
                  nd_label_type = 'ner', 
                  label_transform_model = 'diffusion', 
                  graph_type = 'undirected',
                  pos_model = 1.0, 
                  ner_model = '7classes')

graph_dir = textGraph(punc = punc_tf, stpw = stpw_tf,
                  winSize = graph_win_size,
                  nd_attr_type = node_attr_type,
                  vec_dim = vec_dim, 
                  vec_win_size = vec_win_size, 
                  vec_model = vec_model, 
                  nd_label_type = 'ner', 
                  label_transform_model = 'diffusion', 
                  graph_type = 'directed',
                  pos_model = 1.0, 
                  ner_model = '7classes')

#%% 4. Read Data
graph_undir.fit(labels = labels, texts = texts, corpus_name = corpus_name)
K_ner_undir, Y_ner_undir =graph_undir.computeKernel()

graph_dir.fit(labels = labels, texts = texts, corpus_name = corpus_name + 'dir')
K_ner_dir, Y_ner_dir =graph_dir.computeKernel()

#%% 4.5 TFIDF Kernel
# TFIDF
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()  
X = vectorizer.fit_transform(texts) 

# Kernel
from sklearn.metrics.pairwise import rbf_kernel
K_tfidf = rbf_kernel(X)

#%% Mix kernel
#graphKWeight = 0.5
#K = graphKWeight*K_ner_undir + (1-graphKWeight)*K_tfidf

#%% 5. Bag of words
from modules.representations import bagOfWords
b = bagOfWords(punc = punc_tf, stpw = stpw_tf)
b.fit(labels = labels, texts = texts, corpus_name = corpus_name)
X_b, Y_b = b.toBagOfWords(bow_model = bow_model)

#%% CV: different ratios
from modules.propagationKernel import crossValidate
graphKernelRatioList = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
times_num = 5
folds_num = 5
seeds = [1,2,3,4,5,6,7,8,9,10] # fix the seed
CVs = []
for gkR in graphKernelRatioList:
    K = gkR*K_ner_undir + (1-gkR)*K_tfidf
    CVs.append(crossValidate(Y_ner_undir, K, 'precomputed', seeds, fold = folds_num, times = times_num))
import matplotlib.pyplot as plt
plt.plot(CVs)

#%% CV
#from propagationKernel import crossValidate
#times_num = 5
#folds_num = 5
## seeds = random.sample(range(1,100),10)
#seeds = [1,2,3,4,5,6,7,8,9,10] # fix the seed
#
#print('******** bag of words ({},{}) ********'.format(bow_model, kernel_type))
#crossValidate(Y_b, X_b, kernel_type, seeds, fold = folds_num, times = times_num)
#print('')
#
#print('******** graph of words (ner, undir) ********')
#crossValidate(Y_ner_undir, K_ner_undir, 'precomputed', seeds, fold = folds_num, times = times_num)
#print('')
#
#print('******** graph of words (ner, dir) ********')
#crossValidate(Y_ner_dir, K_ner_dir, 'precomputed', seeds, fold = folds_num, times = times_num)
#print('')
#
#print('******** graph of words (ner, undir+tfidf) ********')
#crossValidate(Y_ner_undir, K, 'precomputed', seeds, fold = folds_num, times = times_num)
#print('')
#
#print('******** graph of words (ner, tfidf) ********')
#crossValidate(Y_ner_undir, K_tfidf, 'precomputed', seeds, fold = folds_num, times = times_num)
#print('')