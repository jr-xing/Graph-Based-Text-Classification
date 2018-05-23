# 2017-3-30
# Yu Sun

from .util import *
from sortedcontainers import SortedSet
from collections import OrderedDict
from collections import deque

#######################################################################
###                             WINDOW                              ###
#######################################################################

class window(object):
	queue = []
	edge_attr = {}

	def __init__(self, winSize, graph_type): #, stpw):
		self.size = winSize
		self.graph_type = graph_type

	# emit DS_A.txt file
	# return a dict
	# windowSize must be greater than one
	def slidingWindow(self, tokens, tokenToID):
		# initialization
		self.queue = deque()
		self.edge_attr = OrderedDict()
		
		#-- the loop --#
		for cur_token in tokens:
			
			# pop()
			q_l = len(self.queue)
			if q_l == 0:
				pass
			elif q_l % self.size == 0: # pop the first element when length of queue reach the window size
				self.queue.popleft()

			# sliding window
			if len(self.queue) == 0:
				pass
			else: # link cur_token to every token in queue
				for token in self.queue:
					token_id = tokenToID[token]
					cur_token_id = tokenToID[cur_token]
					for case in switch(self.graph_type):
						# for directed graph, order indicate the direction of edge
						if case('directed'):
							try:
								self.edge_attr[(token_id,cur_token_id)] += 1 # follows text natural order
							except:
								self.edge_attr[(token_id,cur_token_id)] = 1

						# for undirected graph, create two entries for each edge
						if case('undirected'):
							try:
								self.edge_attr[(token_id,cur_token_id)] += 1 # follows text natural order
							except:
								self.edge_attr[(token_id,cur_token_id)] = 1
							# inverse order of tokens and create the edge
							try:
								self.edge_attr[(cur_token_id,token_id)] += 1 # follows the inverse order
							except:
								self.edge_attr[(cur_token_id,token_id)] = 1

			# append cur_token at last
			self.queue.append(cur_token)
		return self.edge_attr



