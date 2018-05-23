# Graph-Based-Text-Classification
## Configuration
0. This part is based on standard anaconda python environment, where many popular scinentific packages (numpy, matplotlib, scikit-learn, etc.) are already installed.
1. Configure MATLAB Engine
    - Install MATLAB engine for python: [Install MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
    - Add ```/modeules/PropagationKernel/``` to MATLAB path

2. Config Stanford NLP Toolkits
    - Install Java and Stanford following:
        > - [Configuring Stanford Parser and Stanford NER Tagger with NLTK in python on Windows and Linux](https://blog.manash.me/configuring-stanford-parser-and-stanford-ner-tagger-with-nltk-in-python-on-windows-f685483c374a)
        > - [Problem of NLTK with StanfordTokenizer](https://tianyouhu.wordpress.com/2016/09/01/problem-of-nltk-with-stanfordtokenizer/)
    - Edit Stanford and Java path in ```./configs.py```

## Usage
1. Initilize graph variable
    - Usage
        ```python
        # Copy from demo.py
        # Add more comments later
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
        ```

2. Fit data (texts and labels)
    - Please make sure your text has no non-ascii characters to use the Stanford tagger!
    - Usage
        ```python
        # texts is list of texts, e.g. ['I have an apple', 'six-past-ten', 'tenki ii kana']
        # labels is list of labels which has same number elements as tests, e.g. ['Tom', 'Bella', 'Saigo']
        # Now elements in labels can only be strings, will fix later
        # corpus_name is a string for arbitrary name, e.g. 'speaker'
        graph_undir.fit(labels = labels, texts = texts, corpus_name = corpus_name)
        ```

3. Compute graph Kernel
    - Usage
        ```python
        K_ner_undir, Y_ner_undir =graph_undir.computeKernel()
        ```

4. Cross Validation
    - Usage
        ```python
        from modules.propagationKernel import crossValidate
        times_num = 5
        folds_num = 5
        seeds = [1,2,3,4,5,6,7,8,9,10] # fix the seed
        print('******** graph of words (ner, undir) ********')
        crossValidate(Y_ner_undir, K_ner_undir, 'precomputed', seeds, fold = folds_num, times = times_num)
        print('')
        ```