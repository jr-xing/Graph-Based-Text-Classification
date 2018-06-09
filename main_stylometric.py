import pandas as pd
# corpus_name = 'demo_mini'
# corpus_name = 'state_of_the_union_corpus'
corpus_name = 'simpsons_proc_3'

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

# Use part to speed up
# data_raw = data_raw.sample(frac = 0.2)

#%% 2. Extract texts and labels
texts = list(data_raw[texts_header])
labels = list(data_raw[labels_header])

#%% 3. Initialize graph
# Parameters
# **********  Fixed  **********
punc_tf = True
stpw_tf = False
vec_win_size = 10
vec_model = 0

# ********** Hyperparameters **********
graph_win_size = 3
vec_dim = 10
node_attr_type = 'word2vec' # word2vec
bow_model = 'tfidf'
kernel_type = 'rbf' # rbf

# Create graph object
from modules.representations import textGraph
# Undirected graph
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

# Directed graph
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

#%% 4. Fit Data and compute graph kernel
graph_undir.fit(labels = labels, texts = texts, corpus_name = corpus_name + 'undir')
K_ner_undir, Y_ner_undir =graph_undir.computeKernel()

graph_dir.fit(labels = labels, texts = texts, corpus_name = corpus_name + 'dir')
K_ner_dir, Y_ner_dir =graph_dir.computeKernel()

#%% 5. Bag of words
from modules.representations import bagOfWords
b = bagOfWords(punc = punc_tf, stpw = stpw_tf)
b.fit(labels = labels, texts = texts, corpus_name = corpus_name)
X_b, Y_b = b.toBagOfWords(bow_model = bow_model)

#%% 6. Cross Validation
from modules.propagationKernel import crossValidate
times_num = 5
folds_num = 5
seeds = [1,2,3,4,5,6,7,8,9,10] # fix the seed

print('******** bag of words ({},{}) ********'.format(bow_model, kernel_type))
crossValidate(Y_b, X_b, kernel_type, seeds, fold = folds_num, times = times_num)
print('')

print('******** graph of words (ner, undir) ********')
crossValidate(Y_ner_undir, K_ner_undir, 'precomputed', seeds, fold = folds_num, times = times_num)
print('')

print('******** graph of words (ner, dir) ********')
crossValidate(Y_ner_dir, K_ner_dir, 'precomputed', seeds, fold = folds_num, times = times_num)
print('')