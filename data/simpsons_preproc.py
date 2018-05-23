"""
Created on Tue May 01 17:05:04 2018

@author: remussn
"""

#%% 1. Read Raw Data
import numpy as np
import pandas as pd    
filePath = './simpsons_script_lines.csv'
data_script_lines = pd.read_csv(filePath,
                                error_bad_lines=False,
                                warn_bad_lines=False,
                                low_memory=False)

data_true_script_lines = data_script_lines[data_script_lines['speaking_line']=='true']
data_true_script_lines = data_true_script_lines[['character_id', 'raw_character_text', 'normalized_text', 'word_count']]

# Keep only long sentences
data_true_script_lines = data_true_script_lines[data_true_script_lines['word_count']>=10]

#%% 2. Keep only Active Caharcters
commonCount = 3

import collections
counter = collections.Counter(data_true_script_lines['raw_character_text'])

mostCommonCharId = counter.most_common(commonCount)
mostCommonCharId = [i[0] for i in mostCommonCharId]

data_true_script_lines_active = data_true_script_lines[[charID in mostCommonCharId for charID in data_true_script_lines['raw_character_text']]]

#%% 3. Remove non-ascii words and stop words
# Filter out all non-ascii characters
# isascii = lambda s: len(s) == len(s.encode())
# http://hzy3774.iteye.com/blog/2359032
isascii = lambda keyword: all(ord(c) < 128 for c in keyword)
ifRemoveStopWords = False

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
for idx, row in data_true_script_lines_active.iterrows():
    text = row['normalized_text']
    # Remove stop words
    #if ifRemoveStopWords:
    #    text = ' '.join([word for word in text.split() if (word not in stop_words)&(isascii(word))])
    #else:
    if not isascii(text):
        print(text)
        text = ' '.join([word for word in text.split() if isascii(word)])
        print(text)
        data_true_script_lines_active.loc[idx]['normalized_text'] = text


#%% 4. Save data
data_true_script_lines_active[['raw_character_text','normalized_text']].to_csv('./simpsons_proc_{}.csv'.format(commonCount), index=False)
# data_true_script_lines_active[['raw_character_text','normalized_text']].to_csv('./simpsons_proc_{}.txt'.format(commonCount), index=False, encoding='ascii', sep='\t', header=False)
# data.to_csv('./simpsons_proc_5.txt',index = False, encoding='ascii', sep='\t', header=False)

