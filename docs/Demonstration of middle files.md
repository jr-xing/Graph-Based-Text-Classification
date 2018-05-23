## Demonstration of middle files
### exper setting
```python
# *******  Corpus Name  *******
dataset_name = 'demo_mini'

# ********** Hyperparameters **********
graph_win_size = 2
vec_dim = 10
node_attr_type = 'word2vec' # word2vec
bow_model = 'tfidf'
kernel_type = 'rbf' # rbf
graph_type = 'undirected'
```


### file list
- ```<corpus name>_A.txt```
- ```<corpus name>_edge_attributes.txt```
- ```<corpus name>_graph_indicator.txt```
- ```<corpus name>_graph_label_id_dict.txt```
- ```<corpus name>_graph_labels.txt```
- ```<corpus name>_node_labels.txt```
- ```<corpus name>_node_labels_dict.txt```

### demo input file
In the demo input file we have only two samples with 8 words in each:
```python
# input filename: demo_mini.txt
# Content:
Demo	one two one two three four one two
animal	cat dog cat dog panda lion cat dog
```

### generated files
#### ```<corpus name>_A.txt```
- Record the existance of co-occurace between pair of words, in order of words
- Content
    ```python
    # demo_mini_A.txt
    1, 2    # the edge between word 1
            # (i.e. the first word in sample, "one")
            # and word 2 ("two")
    1, 4
    2, 1
    2, 3
    3, 2
    3, 4
    4, 1
    4, 3
    5, 6    # this line and below are words in the second sample
    5, 8
    6, 5
    6, 7
    7, 6
    7, 8
    8, 5
    8, 7
    ```

#### ```<corpus name>_edge_attributes.txt```
- Record the weight of edges between pair of words, i.e. the time of co-occurance, in order of words
- Content
    ```python
    # demo_mini_edge_attributes.txt
    4       # There are 4 co-occurance of word 1 and 2 
            # (i.e "one two" and "two one")
    1
    4
    1
    1
    1
    1
    # begin of second sample
    1
    4
    1
    4
    1
    1
    1
    1
    1
    ```

#### ```<corpus name>_graph_indicator.txt```
<!-- - Indicate the co-occurance (i.e. edge of graph) recorded in ```<corpus name>_A.txt``` and ```<corpus name>_edge_attributes.txt``` are of which sample -->
- Indicate the words (i.e. nodes in graph) are of which sample
- Content
    ```python
    # demo_mini_graph_indicator.txt
    1       # this node belong to the sample 1 
    1
    1
    1
    2       # this node belong to the sample 2
    2
    2
    2
    ```
#### ```<corpus name>_graph_label_id_dict.txt```
- Assign a numerical label to each raw label
- Content
    ```python
    # demo_mini_graph_label_id_dict.txt
    # digital is assigned as class 1 and animal as class 2
    digit: 1
    animal: 2
    ```

#### ```<corpus name>_graph_labels.txt```
- Based on ```<corpus name>_graph_label_id_dict.txt```, record the numercial label of each sample
- Content
    ```python
    # demo_mini_graph_labels.txt
    # digital is assigned as class 1 and animal as class 2
    1
    2
    ```

#### ```<corpus name>_node_labels_dict.txt```
- Node(word) category labels from Stanford tagger. See report for more information
- Content
    ```python
    # demo_mini_node_labels_dict.txt    
    LOCATION: 1
    PERSON: 2
    ORGANIZATION: 3
    MONEY: 4
    PERCENT: 5
    DATE: 6
    TIME: 7
    Unknown: 0
    ```

#### ```<corpus name>_node_labels.txt```
- Based on ```<corpus name>_node_labels_dict.txt```, record the node labels
- Content
    ```python
    # demo_mini_graph_labels.txt
    # All words are "Unknown" category
    0
    0
    0
    0
    0
    0
    0
    0
    ```

#### ```<corpus name>_node_attributes.txt```
- Node attribute vectors. See report for more information
- Content
    ```python
    # demo_mini_node_attributes.txt
    # each row is the 10-dimensional atttibute vector of a node(word). All 8 rows are corresponding to 8 words (e.g. the first row is the attribute vector of word "one").
    0.4273536801338196,-0.17942917346954346,0.04191324859857559,0.14254549145698547,0.05209362879395485,0.6614620089530945,0.23457291722297668,0.29981619119644165,0.13111279904842377,0.4009332060813904
    0.07492895424365997,-0.3840309679508209,-0.3937050402164459,0.11991863697767258,0.35659119486808777,-0.044684041291475296,-0.1267516165971756,-0.5788425803184509,-0.3472048342227936,0.27693310379981995
    -0.3336779475212097,0.3905222415924072,0.5315040349960327,0.1165938749909401,0.021632028743624687,-0.2879568040370941,-0.25435492396354675,0.45563411712646484,0.26517927646636963,0.11855288594961166
    0.4273330569267273,0.29135867953300476,-0.07516822963953018,-0.46197083592414856,-0.27412426471710205,0.5013818144798279,-0.28058716654777527,-0.31019821763038635,-0.04365626722574234,0.10021723806858063
    0.1103881299495697,0.017575709149241447,-0.4946361482143402,0.211720809340477,0.4882161319255829,-0.37709492444992065,0.4416910707950592,-0.2949032485485077,-0.05458969250321388,-0.18005682528018951
    -0.020394979044795036,-0.19861513376235962,-0.5019264817237854,-0.04986637830734253,0.22969463467597961,0.2833802402019501,-0.4825984239578247,0.08907914906740189,0.45513710379600525,-0.3530849814414978
    -0.14723306894302368,-0.4769955277442932,-0.4829683005809784,-0.1454322338104248,0.4483622908592224,0.3379712700843811,-0.12208143621683121,0.1880219280719757,0.2403961420059204,-0.27034997940063477
    0.33442363142967224,-0.3450946509838104,-0.10314047336578369,-0.37411779165267944,0.3595546782016754,0.4195857048034668,-0.3440636396408081,0.2851409614086151,-0.3308408260345459,-0.06321430951356888
    ```
