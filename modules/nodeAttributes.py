# -*- coding: utf-8 -*-
# 2017-4-21
# Yu Sun

#######################################################################
###                          NODE ATTRIBUTES                        ###
#######################################################################

#-- general --#
from .util import *
from sortedcontainers import SortedSet
from collections import OrderedDict

#-- attribute function --#
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import word2vec

import codecs

class nodeAttributes(object):

	def __init__(self, contents, tokenToID_list, corpus_name, punc, stpw):
		self.temp_path 		= '{}_temp.txt'.format(corpus_name)
		self.corpus_name	= corpus_name
		self.contents 		= contents
		self.tokenToID_list = tokenToID_list
		self.punc			= punc
		self.stpw			= stpw
		self.IDToAttr		= OrderedDict()
		self.__helper()

	def __helper(self):
		#temp = open(self.temp_path,"w")  # delete temp.txt when the function is finished
		temp = codecs.open(self.temp_path,"w", encoding='utf-8')  # delete temp.txt when the function is finished

		# remove punc
		if self.punc: 
			self.contents = rmListStringPunctuations(self.contents)
		# remove stopwords
		if self.stpw:
			self.contents = rmListStringStopwords(self.contents)

		for tokenToID in self.tokenToID_list:
			for token in tokenToID.keys():
				temp.write(token + ' ')
			temp.write('\n')
		temp.close()

		# for content in self.contents:
		# 	temp.write(content.lower() + '\n')  # turn it into lowercase in order to be compatiable with self.tokenToID_list
		# temp.close()

		# group words as phrases, such as 'Los' 'Angles' -> 'Los_Angles'
	def __toPhrase(self):
		temp_phrase_path = '{}_temp_phrase.txt'.format(corpus_name)
		word2vec.word2phrase(self.temp_path,
							temp_phrase_path, 
							verbose = True)
		return temp_phrase_path

	def vecAttribute(self, vec_dim, win_size, vec_model):

		#-- train word2vec model --# 
		bin_path = '{}_vectors.bin'.format(self.corpus_name)
		word2vec.word2vec(self.temp_path, bin_path, cbow = vec_model, 
						size = vec_dim, window = win_size, 
						negative = 5, hs = 0, sample = '1e-4', 
						threads = 12, iter_ = 20, 
						min_count = 1, verbose = True)
		mode = word2vec.load(bin_path)

		#-- create IDToAttr --#
		for tokenToID in self.tokenToID_list:
			for token in tokenToID.keys():
				t_id 	= tokenToID[token]
				raw_vec = list(mode[token])
				vec  	= ','.join(map(str, raw_vec))
				self.IDToAttr[t_id] = vec

		#-- delete temp files --#
		os.remove(bin_path)
		os.remove(self.temp_path)
		
		return self.IDToAttr

	def tfidfAttribute(self):
		#-- initializing --# 
		tfidfVectorizer = TfidfVectorizer()
		tfidf_matrix	= tfidfVectorizer.fit_transform(self.contents)
		vocab 			= tfidfVectorizer.vocabulary_   # dict for mapping token to column index

		#-- get corresponding tfidf for each word --#
		# raw order then column order
		for r in range(len(self.contents)):
			tokenToID = self.tokenToID_list[r]
			for token in tokenToID.keys():
				t_id = tokenToID[token]
				c 	 = vocab[token]
				tfidf_t = tfidf_matrix[r,c]
				self.IDToAttr[t_id] = str(tfidf_t)
		return self.IDToAttr

	def bowAttribute(self, bow_model):
		pass
		
		# #-- get bag of words --#
		# for case in switch(bow_model):
		# 	if case('count'):
		# 		countVectorizer = CountVectorizer()
		# 		bow_attr = countVectorizer.fit_transform(self.contents)
		# 		break
		# 	if case('tfidf'):
		# 		tfidfVectorizer = TfidfVectorizer()
		# 		bow_attr = tfidfVectorizer.fit_transform(self.contents)
		# 		break
		# 	if case('hash'):
		# 		hashingVectorizer = HashingVectorizer()
		# 		bow_attr = hashVectorizer.fit_transform(self.contents)
		# 		break
		# 	if case():
		# 		print 'required model is not available'
		# 		quit()

		# #-- create IDToAttr --#
		# for tokenToID in self.tokenToID_list:
		# 	for token in tokenToID.keys():
		# 		t_id 	= tokenToID[token]
		# 		raw_bow = list(bow_attr[t_id,:])
		# 		bow  	= ','.join(map(str, raw_vec))
		# 		self.IDToAttr[t_id] = bow

		# return self.IDToAttr

