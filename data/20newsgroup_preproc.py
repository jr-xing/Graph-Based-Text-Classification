"""
Data preprocessing for 20 news group data


Created on Sun May 20 16:20:52 2018

@author: remussn
"""

#%% 1. Get Raw Data
from sklearn.datasets import fetch_20newsgroups
#from main_tuning_util import getNewsGroupContent
cats = ['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
#newsgroups_train_scripts = getNewsGroupContent(newsgroups_train.data)

#%% 2. Extract Body
import re
stopBeginners = ['From','Subject','Nntp-Posting-Host','Organization',
                     'X-Newsreader','Lines']
#    scriptList = ['From: bil@okcforum.osrhe.edu (Bill Conner)\nSubject: Re: Not the Omni!\nNntp-Posting-Host: okcforum.osrhe.edu\nOrganization: Okcforum Unix Users Group\nX-Newsreader: TIN [version 1.1 PL6]\nLines: 18\n\nCharley Wingate (mangoe@cs.umd.edu) wrote:\n: \n: >> Please enlighten me.  How is omnipotence contradictory?\n: \n: >By definition, all that can occur in the universe is governed by the rules\n: >of nature. Thus god cannot break them. Anything that god does must be allowed\n: >in the rules somewhere. Therefore, omnipotence CANNOT exist! It contradicts\n: >the rules of nature.\n: \n: Obviously, an omnipotent god can change the rules.\n\nWhen you say, "By definition", what exactly is being defined;\ncertainly not omnipotence. You seem to be saying that the "rules of\nnature" are pre-existant somehow, that they not only define nature but\nactually cause it. If that\'s what you mean I\'d like to hear your\nfurther thoughts on the question.\n\nBill\n',
#                  "From: jhwitten@cs.ruu.nl (Jurriaan Wittenberg)\nSubject: Re: Magellan Update - 04/16/93\nOrganization: Utrecht University, Dept. of Computer Science\nKeywords: Magellan, JPL\nLines: 29\n\nIn <19APR199320262420@kelvin.jpl.nasa.gov> baalke@kelvin.jpl.nasa.gov \n(Ron Baalke) writes:\n\n>Forwarded from Doug Griffith, Magellan Project Manager\n>\n>                        MAGELLAN STATUS REPORT\n>                            April 16, 1993\n>\n>\n>2.  Magellan has completed 7225 orbits of Venus and is now 39 days from\n>the end of Cycle-4 and the start of the Transition Experiment.\nSorry I think I missed a bit of info on this Transition Experiment. What is it?\n\n>4.  On Monday morning, April 19, the moon will occult Venus and\n>interrupt the tracking of Magellan for about 68 minutes.\nWill this mean a loss of data or will the Magellan transmit data later on ??\n\nBTW: When will NASA cut off the connection with Magellan?? Not that I am\nlooking forward to that day but I am just curious. I believe it had something\nto do with the funding from the goverment (or rather _NO_ funding :-)\n\nok that's it for now. See you guys around,\nJurriaan.\n \n-- \n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n|----=|=-<- - - - - - JHWITTEN@CS.RUU.NL- - - - - - - - - - - - ->-=|=----|\n|----=|=-<-Jurriaan Wittenberg- - -Department of ComputerScience->-=|=----|\n|____/|\\_________Utrecht_________________The Netherlands___________/|\\____|\n"]
scriptListProced = []
for script in newsgroups_train.data:        
    content = ''
    for line in script.split('\n'):
        if (len(line)>1)&(not line.startswith(tuple(stopBeginners))):
            content = content + re.sub(r'[^\w\s]','',line) + '\n'
    scriptListProced.append(content)
    
#%% 3. Remove rare words
import itertools, collections
words = list(itertools.chain(*[script.split() for script in scriptListProced]))
wordsSet = set(words)
wordCounter = collections.Counter(words)
notRareWords = [word for word in words if wordCounter[word]>2]