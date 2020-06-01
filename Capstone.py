# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import pdb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json as js
import nltk
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer,SnowballStemmer
from nltk.stem.porter import *
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# %% [code]
# clean up data
stemmer = SnowballStemmer('english')
vocab_size = 10000


"""
extract verb from text
"""
def lemmatize_stemming(text):
    return stemmer.stem(text)
    #return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


"""
clear trivial words such as a, an, the and so forth
"""
def preprocess_stem_clean(text):
    result = []
    """
    Convert a document into a list of tokens.
    This lowercases, tokenizes, de-accents (optional). – 
    the output are final tokens = unicode strings, that won’t be processed any further.
    """
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(lemmatize_stemming(token))
    return result


# %% [code]
def preprocess_data():
    i = 0
    # word related to diagnositics and surveillance
    related_words = ['diagnostic', 'diagnostics', 'symptomatic', 'diagnosing', 'diagnosis', 'clinical',
                     'diagnose', 'diagnoses', 'detection', 'screening', 'analytical', 'assessment',
                    'prognosis', 'surveillance', 'monitoring', 'reconnaissance']
    corona_pos = []
    corona_pos_all_text = []
    label=[]

    keywords_cnt = 0
    keywords_list = []

    """
    walk through all the files under specific directory
    """
    for dirname, _, filenames in os.walk('C:/PythonWorkspace/document_parses/pdf_json'):
        for filename in filenames:
            #print(os.path.join(dirname, filename))

            topic_related = False
            if i % 1000 == 0:
                print ("Working (number %d)..." % i)

            """
            only load json files
            """
            if filename.split(".")[-1] == "json":

                f = open(os.path.join(dirname, filename))
                j = js.load(f)
                f.close()

                """
                some articles contain abstract while others not 
                """
                try:
                    abstract_text = ' '.join([x['text'] for x in j['abstract']])
                except:
                    abstract_text = ""

                """
                extract all the keywords
                """
                tmp=''
                # d represents each individual dictionary
                # j['body_text'] is a python list which contains many dictionaries
                for d in j['body_text']:
                    if d['section']=='Keywords':
                        tmp+=d['text']
                        keywords_cnt+=1
                        keywords_list.append(tmp)


                """
                body text and abstract consist the whole body text
                """
                body_text = ' '.join(x['text'] for x in j['body_text'])
                body_text += " " + abstract_text

                for related_word in related_words:
                    if related_word in body_text:
                        topic_related = True
                        break
                if topic_related:
                    i += 1
                    corona_pos_all_text.append(body_text)
    return corona_pos_all_text, corona_pos, label, keywords_cnt, keywords_list


"""
we use wordCloud library to show the most frequent words in each cluster 
"""
def word_cloud_advanced(corona_pos_all_text):

    """
    wordCloud only accepts string as input
    """
    stopwords=set(STOPWORDS)
    res = ''
    for text in corona_pos_all_text:
        for word in text:
            res += word + ' '

    # generate wordcloud image
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(res)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()



corona_pos_all_text, corona_pos, label, keywords_cnt, keywords_list = preprocess_data()
print('The total number of total documents is '+str(len(corona_pos_all_text)))
keywords_all_text = [preprocess_stem_clean(x) for x in keywords_list]
word_cloud_advanced(keywords_all_text)
corona_pos_all_text = [len(preprocess_stem_clean(x)) for x in corona_pos_all_text]
plt.bar([i for i in range(len(corona_pos_all_text))],corona_pos_all_text)
plt.xlabel('Data')
plt.ylabel('Words of each article')
plt.show()



