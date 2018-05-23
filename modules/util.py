import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
################ Preprocessing Functions #################

def isStpw(token):
    sw = stopwords.words('english')
    if token.lower() in sw:
        return True
    else:
        return False

def rmPunctuations(tokens):
    punc = string.punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    return tokens

def rmStopwords(tokens):
    sw = stopwords.words('english')
    tokens = [token for token in tokens if token.lower() not in sw]
    return tokens

def rmListStopwords(list_of_tokens):
	new = []
	for tokens in list_of_tokens:
		new.append(rmStopwords(tokens))
	return new

def rmListPunctuations(list_of_tokens):
	new = []
	for tokens in list_of_tokens:
		new.append(rmPunctuations(tokens))
	return new

# replace punc with whitespace. this won't hurt tokenizing
def rmStringPunctuations(text):
    new = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    return new
    # this way is more fast
    # return text.translate(None, string.punctuation)

# text without punctuations
def rmStringStopwords(text):
    tokens = rmStopwords(word_tokenize(text))
    sent = ' '.join(token for token in tokens)
    return sent

def rmStringShortWords(text, min):
    old_tokens = word_tokenize(text)
    new_tokens = [token for token in old_tokens if len(token) >= min]
    sent = ' '.join(token for token in new_tokens)
    return sent

def rmListStringPunctuations(text_list):
    new = []
    for text in text_list:
        new.append(rmStringPunctuations(text))
    return new

def rmListStringStopwords(text_list):
    new = []
    for text in text_list:
        new.append(rmStringStopwords(text))
    return new

def rmListStringShortWords(text_list, min):
    new = []
    for text in text_list:
        new.append(rmStringShortWords(text, min))
    return new

def rmListStringShortPunc(text_list, min):
    temp = rmListStringPunctuations(text_list)
    new = rmListStringShortWords(temp, min)
    return new
    
################### Node Label Functions ###################
from collections import Counter

def mostCommon(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


################### Ordered Set ###################

import collections

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

################### Switch Class ###################

class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

