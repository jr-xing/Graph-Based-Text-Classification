# -*- coding: utf-8 -*-
# 2017-4-21
# Yu Sun

import os
import numpy as np
from .util import *
from sortedcontainers import SortedSet
from collections import OrderedDict
from nltk.tokenize import word_tokenize

    
#######################################################################
#######                                                         #######
#######                    ALL REPRESENTATIONS                  #######
#######                                                         #######
#######################################################################

# input format:
#	label + '\t' + text
from configs import configs
debug_print = lambda s: print(s)
class textRepresentations(object):

    # how to pass the parameter to children class, one thought: import parse() in the father class.
    # preprocessing with father class
    #def __init__(self, in_path, corpus_name):
    def __init__(self):
        # initialize parameters
        self.labels 	 = [] # list of label of each datapoint (lowercased)
        self.contents 	 = [] # list of contents of each datapoints (original raw data)
        self.IDTolabel   = OrderedDict() # dic mapping id to unique label, keep in order
        self.labelToID   = OrderedDict() # dic mapping label to unique id, keep in order        
        self.corpus_name = ''

    # order of labels is kept in labels
    def getLabelList(self):
        return self.labelToID.keys()

    def getLabelIdxList(self):
        return [self.labelToID[label] for label in self.labels]
        # labelIdxList = []
        # for label in self.labels:
        # 	labelIdxList.append(self.labelToID[label])
        # return labelIdxList

    # order of contents is kept in self.contents
    def getContentList(self):
        return self.contents


###############################################################
#####                         GRAPH                       #####
###############################################################

from .window import window
from .nodeLabel import nodeLabels
from .nodeAttributes import nodeAttributes
from .propagationKernel import propKernel
import pandas as pd
# from configs import configs

class textGraph(textRepresentations):
    
    'class Graph, represent text as graph, successor of class representation.'
    
    def __init__(self, winSize, nd_attr_type, vec_dim, vec_win_size, vec_model, nd_label_type, label_transform_model, temp_data_path = './data/graph_temp', min_wl = 2, punc = True, stpw = True, ner_model = '7classes', pos_model = 1.0, graph_type = 'undirected'):
        super(textGraph, self).__init__()
        self.punc 	= punc
        self.stpw 	= stpw
        self.min_wl = min_wl
        self.all_words = OrderedSet() # all unique
        self.tokens_list 	= [] # list of list of tokens of each datapoint, operation on punctuations only.
        self.tokenToID_list = [] # list of dictionaries, each dictionary for one datapoint, operation on punctuations only.
        self.K 				= None
        self.Y 				= None
        self.temp_data_path = temp_data_path
        
        self.winSize = winSize
        self.nd_attr_type = nd_attr_type
        self.vec_dim = vec_dim
        self.vec_win_size = vec_win_size
        self.vec_model = vec_model
        self.nd_label_type = nd_label_type
        self.label_transform_model = label_transform_model
        self.ner_model = ner_model
        self.pos_model = pos_model
        self.graph_type = graph_type
        
        # graph info        
        self.graph_edges = {'nodes':[], 'attributes':[], 'graph_indicators':[]}   # vertices, attributes, graph_indicators
        self.graph_nodes = {'labels':[], 'attributes':[]}   # labels, attributes
    
    def read(self, texts, labels, corpus_name = 'graph'):
        # read and store labels and texts
        # texts and labels should be python lists
        self.corpus_name = corpus_name
        # print('CORPUS NAME:\n'+self.corpus_name)
        
        if len(texts) != len(labels):
            print('Numbers of labels and texts should be the same!')
            quit()
        labels_set = OrderedSet()  # a list that store all different labels. keep order!
        for idx in range(len(texts)):
            label = labels[idx]
            content = texts[idx]
            if len(content)<1:  continue
            self.contents.append(content) 	   # no preprocessing excution on contents
            self.labels.append(label)
            labels_set.add(label)
        
        #-- give id to each label --#
        label_id = 1
        for ele in labels_set:
            self.labelToID[ele] = label_id
            self.IDTolabel[str(label_id)]=ele
            label_id = label_id + 1
        self.__helper()
    
    def __helper(self):
        # remove punc
        # debug_print(self.contents)
        contents = self.contents
        if self.punc: 
            #contents = rmListStringPunctuations(self.contents)
            contents = rmListStringPunctuations(contents)

        # remove stopwords
        if self.stpw:
            contents = rmListStringStopwords(contents)
        
        # remove word that its length less than min_wl
        contents = rmListStringShortWords(contents, self.min_wl)                

        t_id = 1 # token id starts at 1
        for content in contents:
            tokenToID = OrderedDict()
            words = OrderedSet()
            tokens = word_tokenize(content.lower())  # uppercase --> lowercase
            self.tokens_list.append(tokens)
            # unique words
            for token in tokens:
                words.add(token)
                self.all_words.add(token)
            # generate dictionary for each sentence
            for word in words:
                tokenToID[word] = t_id
                t_id = t_id + 1
            # add to tokenToID dictionary
            self.tokenToID_list.append(tokenToID)

    # output DS_node_labels.txt
    def __getNodesLabels(self, nd_label_type, ner_model, pos_model):
        # Stanford PosTagger()
        #-- get labels -- #
        getLabel = nodeLabels(self.contents, self.tokenToID_list, self.min_wl, self.punc, self.stpw)

        # not remove stopwords in IDToTag
        for case in switch(nd_label_type):
            if case('pos'):
                IDToTag, node_labelToID = getLabel.posTag(pos_model)
                break
            if case('ner'):
                IDToTag, node_labelToID = getLabel.nerTag(ner_model)
                break
            if case('word'):
                IDToTag, node_labelToID = getLabel.wordTag(self.all_words)

            if case('naive'):
                IDToTag, node_labelToID = getLabel.naive()
                break
            if case('none'):
                print('No {}_node_labels.txt output'.format(self.corpus_name))
#                try:
#                    os.remove(ds_node_labels_path)
#                except:
#                    pass
                return
            if case():
                print('nd_label_type is not available')
                quit()

        #-- label dictionary --#
        #-- e.g. cat:1, dog:2--#
        print('An unique ID will be assigned to each node label. The dictionary will be exported as a text file in the database folder.\nNode_label Dictionary:')
        for key in node_labelToID.keys():
            l_id = node_labelToID[key]
            print('\t' + str(key) + ': ' + str(l_id))        

        #-- node_label files --#
        #       key->   label
        # e.g. 'one' -> 0(Unknown)
        for key in IDToTag.keys():
            label = IDToTag[key]
            self.graph_nodes['labels'].append(label)
            

    # output DS_node_attributes.txt
    def __getNodesAttributes(self, nd_attr_type, vec_dim, vec_win_size, vec_model):
        #-- intialize path --#
        # ds_node_attr_path = self.temp_data_path+'/{}/{}_node_attributes.txt'.format(self.corpus_name,self.corpus_name)

        #-- get labels --#
        getAtrr = nodeAttributes(self.contents, self.tokenToID_list, self.corpus_name, self.punc, self.stpw)
        # switch
        for case in switch(nd_attr_type):
            if case('word2vec'):
                IDToTag = getAtrr.vecAttribute(vec_dim, vec_win_size, vec_model)
                break
            if case('tfidf'):
                IDToTag = getAtrr.tfidfAttribute()
                break
            if case('none'):
                print('No {}_node_attributes.txt output'.format(self.corpus_name))
#                try:
#                    os.remove(ds_node_attr_path)  # remove previous attribute files
#                except:
#                    pass
                return
            if case():
                print('nd_attr_type is not available')
                quit()

        #-- node attributes files --#
        print('Use {} model to compute each word a unique embedding.'.format(nd_attr_type))
        import numpy as np
        for key in IDToTag.keys():
            attr = IDToTag[key]            
            attrlist = [np.float64(a) for a in attr.split(',')]
            self.graph_nodes['attributes'].append(attrlist)
    
    # output DS_A.txt, DS_edge_attr.txt, DS_grpah_indicator.txt
    # all files related to edges
    def __getEdges(self, winSize, graph_type):
        #-- generate window --#
        win = window(winSize, graph_type)
        num_graph = 0
        # search edge info by nodes(words)
        for i in range(len(self.tokens_list)):
            num_graph = num_graph + 1 # graph num starts at 1
            edge_attr = win.slidingWindow(self.tokens_list[i], self.tokenToID_list[i]) # use sliding window
            for edge in sorted(edge_attr.keys()):
                self.graph_edges['nodes'].append([edge[0],edge[1]])
                self.graph_edges['attributes'].append(edge_attr[edge])
            for j in sorted(self.tokenToID_list[i].values()):
                self.graph_edges['graph_indicators'].append(num_graph)


    # return numpy arrary variables
    # 	K:	Kernel Matrix
    # 	Y:	Label
    # use attr_diff
    def toGraph(self):
        print('')
        print('********************')
        print('* toGraph() starts *')
        print('********************')        

        #-- output graph_label_id_dic txt file --# 
        print('GRAPH EDGES: ')
        print('An unique ID will be assigned to each graph label. The dictionary will be exported as a text file in the database folder.\nGraph_label Dictionary:')

        for key,item in self.labelToID.items():
            print('\t' + key + ': ' + str(item))

        #-- output ds_graph_labels.txt --#

        #-- output ds_a_path, ds_edge_attr_path, ds_graph_indicator --#
        self.__getEdges(self.winSize, self.graph_type)

        print('NODE LABELS: ')
        #-- output labels of nodes --#
        self.__getNodesLabels(self.nd_label_type, self.ner_model, self.pos_model)

        print('NODE ATTRIBUTES: ')
        #-- output attributes of nodes --#
        self.__getNodesAttributes(self.nd_attr_type, self.vec_win_size, self.vec_dim, self.vec_model) 
        
        # Save as .mat
        # Check & create result folder
        if not os.path.exists(self.temp_data_path):
            os.mkdir(self.temp_data_path)
        
#        if not os.path.exists(self.temp_data_path+'/{}/'.format(self.corpus_name)):
#            os.mkdir(self.temp_data_path+'/{}/'.format(self.corpus_name))

        # Save
        import numpy as np
        from scipy.io import savemat
        from scipy.sparse import csr_matrix
        rows = [idxs[0]-1 for idxs in self.graph_edges['nodes']]  # node order starts from 1 while matrix index starts from 0
        cols = [idxs[1]-1 for idxs in self.graph_edges['nodes']]
        adj_mat = csr_matrix((self.graph_edges['attributes'], (rows, cols)))#.toarray()
        #savemat(self.temp_data_path+'/{}/{}.mat'.format(self.corpus_name, self.corpus_name), {'A':adj_mat, 'graph_ind':self.graph_edges['graph_indicators'], 'node_labels':self.graph_nodes['labels'], 'node_attr': self.graph_nodes['attributes']})
        savemat(self.temp_data_path+'/{}.mat'.format(self.corpus_name, self.corpus_name), {'A':adj_mat, 'graph_ind':self.graph_edges['graph_indicators'], 'node_labels':self.graph_nodes['labels'], 'node_attr': self.graph_nodes['attributes']})                
        

    def __computeKernel(self, recomputeKernel=True):
        #-- get kernel --#
        temp 	  = [str(self.corpus_name), str(self.punc), str(self.stpw), 
                    str(self.winSize),   str(self.nd_attr_type),  str(self.vec_dim), str(self.vec_win_size), 
                    str(self.vec_model), str(self.nd_label_type), str(self.label_transform_model), 
                    str(self.ner_model), str(self.pos_model),     str(self.graph_type)]

        file_name = self.temp_data_path + '/' + '_'.join(temp) + '_Kernel'

        if not recomputeKernel:
        # If use batch of data with same name, please recompute kernel
            try: # load kernel
                self.K = np.load(file_name + '.npy')
                print('KERNEL LOADED')
    
            except: # compute kernel
                print('KERNEL COMPUTING...')
                #self.K = propKernel(path_to_dataset = './data', 
                self.K = propKernel(path_to_dataset = self.temp_data_path, 
                                    dataset_name = self.corpus_name,
                                    label_transform_model = self.label_transform_model) 
                print('KERNEL SAVING...')
                np.save(file_name, self.K)
        else:
            print('KERNEL COMPUTING...')
            self.K = propKernel(path_to_dataset = self.temp_data_path, 
                                dataset_name = self.corpus_name,
                                label_transform_model = self.label_transform_model) 
            # print('KERNEL SAVING...')
            # np.save(file_name, self.K)

        L = self.getLabelIdxList()
        self.Y = np.array(L)

        return self.K, self.Y # [self.labelToID[label] for label in self.labels] # as Y
    
    # scikit-learn style
    # fit(X,y), predict(X), score(X,y)
    def fit(self, texts, labels, corpus_name = 'graph'):
        # Read raw data, generate graph
        self.read(texts, labels, corpus_name = corpus_name)        
        self.toGraph()
    
    def fit_transform(self, texts, labels, corpus_name = 'graph'):
        # Read raw data, generate graph
        self.read(texts, labels, corpus_name = corpus_name)
        self.toGraph()
        return {'adj_mat': self.adj_mat, 'nodes':self.graph_nodes}
    
    def computeKernel(self):
        # Read text and label, compute kernel, fit svm        
        K, Y_ID = self.__computeKernel()
        self.K = K
        self.normalizeKernel()
        return K, Y_ID
#        Y_label = []
#        for idx in range(len(Y_ID)):
#            Y_label.append(self.IDTolabel[str(Y_ID[idx])])
#        return K, Y_label
        
    def normalizeKernel(self):
        import numpy as np
        num_nodes = len(self.K)
        for iIdx in range(num_nodes):
            for jIdx in range(num_nodes):
                self.K[iIdx,jIdx] = self.K[iIdx,jIdx]/np.sqrt(self.K[iIdx,iIdx]*self.K[jIdx,jIdx])    
    
    # get K
    def get_K(self):
        return self.K

    def get_Y(self):
        return self.Y

########################################################
#####                 BAG OF WORDS                 #####
########################################################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class bagOfWords(textRepresentations):
    'class bagOfWords, represents text as bag of words, successor of class representation.'	

    # def __init__(self, in_path, corpus_name, punc = True, stpw = True):
        # super(bagOfWords, self).__init__(in_path, corpus_name)
    def __init__(self, punc = True, stpw = True):
        super(bagOfWords, self).__init__()
        self.punc = punc
        self.stpw = stpw
        self.X 	  = None
        self.Y    = None
    
    def read(self, texts, labels, corpus_name = 'BOG'):
        # read and store labels and texts
        # texts and labels should be python lists
        self.corpus_name = corpus_name
        
        if len(texts) != len(labels):
            print('Numbers of labels and texts should be the same!')
            quit()
        labels_set = OrderedSet()  # a list that store all different labels. keep order!
        for idx in range(len(texts)):
            label = labels[idx]
            content = texts[idx]
            if len(content)<1:  continue
            self.contents.append(content) 	   # no preprocessing excution on contents
            self.labels.append(label)
            labels_set.add(label)
        
        #-- give id to each label --#
        label_id = 1
        for ele in labels_set:
            self.labelToID[ele] = label_id
            self.IDTolabel[str(label_id)]=ele
            label_id = label_id + 1
        # self.__helper()
    
    def __helper(self):
        # remove punc
        if self.punc: 
            self.contents = rmListStringPunctuations(self.contents)
        # remove stopwords
        if self.stpw:
            self.contents = rmListStringStopwords(self.contents)
    
    
    def fit(self, texts, labels, corpus_name = 'graph'):
        # Read raw data, generate graph
        self.read(texts, labels, corpus_name = corpus_name)
        self.toBagOfWords()    
    

    def toBagOfWords(self, bow_model = 'count'):
        print('')
        print('*************************')
        print('* toBagOfWords() starts *')
        print('*************************')
        #-- select --#
        for case in switch(bow_model):
            if case('count'):
                print('COMPUTING COUNTS')
                countVectorizer = CountVectorizer()
                self.X = countVectorizer.fit_transform(self.contents)
                break
            if case('tfidf'):
                print('COMPUTING TF-IDF')
                tfidfVectorizer = TfidfVectorizer()
                self.X = tfidfVectorizer.fit_transform(self.contents)
                break
            if case('hash'):
                print('COMPUTING HASHING')
                hashingVectorizer = HashingVectorizer()
                self.X = hashVectorizer.fit_transform(self.contents)
                break
            if case():
                print('required model is not available')
                quit()

        #-- get X,Y --#
        L = self.getLabelIdxList()
        # print(L)
        self.Y = np.array(L)

        return self.X, self.Y # [self.labelToID[label] for label in self.labels] # as Y

    def get_X():
        return self.X

    def get_Y(self):
        return self.Y

########################################################
#####                   VECTOR                     #####
########################################################

import word2vec

class vector(textRepresentations):
    'class doc2vec, represents text as vector, successor of class representation.'

    def __init__(self, in_path, corpus_name, punc = True, stpw = True):
        super(vector, self).__init__(in_path, corpus_name)
        self.temp_path = '{}_temp.txt'.format(corpus_name)
        self.punc 	   = punc
        self.stpw 	   = stpw
        self.X 		   = []
        self.Y 		   = None
        self.__helper()

    def __helper(self):
        temp = open(self.temp_path,"w")  # delete temp.txt when the function is finished
        
        # remove punc
        if self.punc: 
            self.contents = rmListStringPunctuations(self.contents)
        # remove stopwords
        if self.stpw:
            self.contents = rmListStringStopwords(self.contents)

        c_id = 1		
        for content in self.contents:
            temp.write(str(c_id) + " " + content.lower() + '\n')
            c_id += 1
        temp.close()

    def toVector(self, vec_dim, win_size, vec_model):
        print('')
        print('*********************')
        print('* toVector() starts *')
        print('*********************')
        # cbow = 0/1
        # size = vec_dim
        # window = win_size
        bin_path = '{}_vectors.bin'.format(self.corpus_name)
        word2vec.doc2vec(self.temp_path, bin_path, cbow = vec_model, 
                        size = vec_dim, window = win_size, 
                        negative = 5, hs = 0, sample = '1e-4', 
                        threads = 12, iter_ = 20, 
                        min_count = 1, verbose = True)
        mode = word2vec.load(bin_path)
        for c_id in range(len(self.contents)):
            self.X.append(mode[str(c_id+1)])
        os.remove(bin_path)
        os.remove(self.temp_path)

        #-- get X,Y --#
        L = self.getLabelIdxList()
        self.Y = np.array(L)

        return np.array(self.X), self.Y

    def get_X(self):
        return np.array(self.X)

    def get_Y(self):
        return self.Y
