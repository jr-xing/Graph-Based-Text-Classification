# 2017-4-21
# Yu Sun

# contains node labels that we need

# WORD CLASS DICTIONARY
# CC - Coordinating conjunction
# CD - Cardinal number
# DT - Determiner
# EX - Existential there
# FW - Foreign word
# IN - Preposition or subordinating conjunction
# JJ - Adjective
# JJR - Adjective, comparative
# JJS - Adjective, superlative
# LS - List item marker
# MD - Modal
# NN - Noun, singular or mass
# NNS - Noun, plural
# NNP - Proper noun, singular
# NNPS - Proper noun, plural
# PDT - Predeterminer
# POS - Possessive ending
# PRP - Personal pronoun
# PRP$ - Possessive pronoun (prolog version PRP-S)
# RB - Adverb
# RBR - Adverb, comparative
# RBS - Adverb, superlative
# RP - Particle
# SYM - Symbol
# TO - to
# UH - Interjection
# VB - Verb, base form
# VBD - Verb, past tense
# VBG - Verb, gerund or present participle
# VBN - Verb, past participle
# VBP - Verb, non-3rd person singular present
# VBZ - Verb, 3rd person singular present
# WDT - Wh-determiner
# WP - Wh-pronoun
# WP$ - Possessive wh-pronoun (prolog version WP-S)
# WRB - Wh-adverb


#######################################################################
###                           NODE LABELS                           ###
#######################################################################

from .util import *
from nltk.tokenize import sent_tokenize
from sortedcontainers import SortedSet
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag_sents
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
from configs import configs
debug_print = lambda s: print(s)

class nodeLabels(object):
    # Framework:
        # 0. __init__() initialize all needed variables
        # 1. helper() generates all needed variables
        # 2. Correspondent tag function generates node_tags(node_labels)

    # contents = list of text
    # tokenToID_list = list of dictionaries
    # how to define self. variables
    # for example, we can remove self.contents
    def __init__(self, contents, tokenToID_list, min_wl, punc, stpw):
        self.contents 		= contents
        self.tokenToID_list = tokenToID_list
        self.maxL_id 		= 0
        self.punc 			= punc
        self.stpw 			= stpw
        self.min_wl			= min_wl
        self.node_labelToID = OrderedDict()  # label to id
        self.IDToTag 		= OrderedDict()	# token id to tag id
        self.content_sents_list = [] # list of list of content sentences
        self.content_sents_num 	= []  # list of num of sentences in each datapoint
        self.__helper()
        self.stanford_path = configs['stanford']['path'] 

    # help generate a list of sentences of each content
    # list of contents' sentences
    # create a list that contains num of sentences in each content
    def __helper(self):
        for content in self.contents:
            sentences = sent_tokenize(content) # split content into sentences. keep punctuations
            self.content_sents_num.append(len(sentences))  	# list of contnet sentences
            for sentence in sentences:
                if self.punc: 		# consistent with graph
                    sentence = rmStringPunctuations(sentence)
                tokens = word_tokenize(sentence)
                self.content_sents_list.append(tokens)

    def __tagFilter(self, rate):
        #-- stats --#
        dic = {}
        for label in self.IDToTag.values():
            try:
                dic[label] += 1
            except:
                dic[label] = 1
        s = sorted(dic, key = dic.__getitem__, reverse = True)
        
        #-- select --#
        cover = 0
        index = 0
        selected = set()
        total = len(self.IDToTag.keys())
        for i in range(len(s)):
            cover += dic[s[i]]
            cover_rate = float(cover)/float(total)
            selected.add(s[i])
            if cover_rate >= rate:
                index = i
                break

        #-- filter --#
        for key in self.IDToTag:
            label = self.IDToTag[key]
            if label not in selected:
                self.IDToTag[key] = self.maxL_id

    # PosTagger Helper
    def __asgnPosID(self):
        pos = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
             'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 
             'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO',
             'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
             'WP', 'WP$', 'WRB']
        l_id = 1 # label id
        for p in pos:
            self.node_labelToID[p] = l_id
            l_id += 1

    # NerTagger Helper
    def __asgnNerID(self, ner_model):
        for case in switch(ner_model):
            if case('3classes'):
                ner = ['LOCATION','PERSON','ORGANIZATION']
                break
            if case('4classes'):
                ner = ['LOCATION','PERSON','ORGANIZATION','MISC']
                break
            if case('7classes'):
                ner = ['LOCATION','PERSON','ORGANIZATION',
                        'MONEY','PERCENT','DATE','TIME']
                break
            if case():
                print('No such ner_model available.')
                quit()

        l_id = 1 # label id starts at 1
        for p in ner:
            self.node_labelToID[p] = l_id
            l_id += 1


    # 'naive'
    def naive(self):
        return self.IDToTag, self.node_labelToID

    # 'word'
    # currently useless but keep it for further experiments
    # just output word itself.
    def wordTag(self, all_words):
        w_id = 1
        for word in all_words:
            self.node_labelToID[word] = w_id
            w_id = w_id + 1

        for i in range(len(self.contents)):
            content = sent_tokenize(self.contents[i]) # split content into sentences. keep punctuations
            tokenToID = self.tokenToID_list[i]
            for sentence in content:
                if self.punc: # consistent with graph
                    sentence = rmStringPunctuations(sentence)
                if self.stpw:
                    sentence = rmStringStopwords(sentence)	
                tokens = word_tokenize(sentence)
                for token in tokens:
                    t_id = tokenToID[token]
                    self.IDToTag[t_id] = str(self.node_labelToID[token])

        return self.IDToTag, self.node_labelToID


    # 'pos'
    # return pos_tag for each sentence
    # select tag with highest frequency
    def posTag(self, pos_model):
        #-- initialize --#
        self.__asgnPosID()

        # path_to_jar = '/Users/SunYu/nltk_data/stanford-postagger-2016-10-31/stanford-postagger.jar'
        # path_to_models = '/Users/SunYu/nltk_data/stanford-postagger-2016-10-31/models/english-bidirectional-distsim.tagger'
        # snt = StanfordPOSTagger(path_to_models, path_to_jar)

        #--- tag ---#
        # tagged_list = snt.tag_sents(self.content_sents_list)
        tagged_list = pos_tag_sents(self.content_sents_list)

        #--- map node label to label id ---#
        fence_h = 0
        fence_t = 0
        for i in range(len(self.content_sents_num)):
            num = self.content_sents_num[i]
            fence_t = fence_h + num
            tokenToID = self.tokenToID_list[i]
            taggeds = tagged_list[fence_h:fence_t]
            for tagged in taggeds:   # create dict
                for ele in tagged:
                    token = ele[0].lower()   # uppercase --> lowercase
                    # ** remove stopwords **
                    if self.stpw and isStpw(token) or len(token) < self.min_wl:
                        continue
                    t_id = tokenToID[token]
                    try:
                        self.IDToTag[t_id] += ',' + str(ele[1])
                    except:
                        self.IDToTag[t_id] = str(ele[1])    # make sure it is string
            fence_h = fence_t	# updata fence

        #-- select most common tags --#
        for key in self.IDToTag.keys():
            t = self.IDToTag[key]
            tags = t.split(',')
            try:
                label = self.node_labelToID[mostCommon(tags)]
            except:
                label = self.maxL_id
            self.IDToTag[key] = label

        #-- tag filtering --#
        self.__tagFilter(pos_model)

        self.node_labelToID['Unknown'] = self.maxL_id # add unknown id to node_labelToID dictionary

        return self.IDToTag, self.node_labelToID

    # availabel models:
    # 3classes / 4-classes / 7-classes
    def nerTag(self, ner_model):
        #-- initialize --#
        self.__asgnNerID(ner_model)
        
        print('*************************')
        print(self.stanford_path)
        print('*************************')

        #-- model selection --#
        for case in switch(ner_model):
            if case('3classes'):
                path_to_models = self.stanford_path+'/classifiers/english.all.3class.distsim.crf.ser.gz'
                break
            if case('4classes'):
                path_to_models = self.stanford_path+'/classifiers/english.conll.4class.distsim.crf.ser.gz'
                break
            if case('7classes'):
                #path_to_models = '/Users/SunYu/nltk_data/stanford-ner-2016-10-31/classifiers/english.muc.7class.distsim.crf.ser.gz'
                path_to_models = self.stanford_path+'/classifiers/english.muc.7class.distsim.crf.ser.gz'
                break
            if case():
                print('No such ner_model available.')
                sys.exit()
                
        snt = StanfordNERTagger(path_to_models, self.stanford_path+'\\stanford-ner.jar')

        #--- tag ---#
        tagged_list = snt.tag_sents(self.content_sents_list) # list of tagged contents

        #--- map token id to label id ---#
        fence_h = 0
        fence_t = 0
        for i in range(len(self.content_sents_num)):
            num = self.content_sents_num[i]
            fence_t = fence_h + num
            # debug_print(self.tokenToID_list)
            tokenToID = self.tokenToID_list[i]
            taggeds = tagged_list[fence_h:fence_t]
            for tagged in taggeds:        # create dict
                for ele in tagged:
                    #token = ele[0].encode().lower()  # uppercase --> lowercase
                    token = ele[0].lower()  # uppercase --> lowercase
                    # ** remove stopwords **
                    if self.stpw and isStpw(token) or len(token) < self.min_wl:
                        continue
                    t_id = tokenToID[token]
                    try:
                        self.IDToTag[t_id] += ','+ str(ele[1])
                        # print(self.IDToTag[t_id])
                    except:
                        self.IDToTag[t_id] = str(ele[1])    # make sure it is string
                        # print(self.IDToTag[t_id])
            fence_h = fence_t	# updata fence


        #-- select most common tags --#
        for key in self.IDToTag.keys():
            t = self.IDToTag[key]
            tags = t.split(',')
            # print(tags)
            try:
                label = self.node_labelToID[mostCommon(tags)]
            except:
                label = self.maxL_id
            self.IDToTag[key] = label

        self.node_labelToID['Unknown'] = self.maxL_id # add unknown id to node_labelToID dictionary
        return self.IDToTag, self.node_labelToID
        

