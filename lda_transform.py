# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


import os
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


from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

"""
There is no fixed solution for anaconda pyLDAvis installation deprecated warnings
"""
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import csv
import xlrd
import xlsxwriter


import random
import copy
import collections
import re
from operator import itemgetter
import seaborn as sns


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn import mixture



import tensorflow as tf
import tensorflow_hub as hub
import bert





"""
dimension reduction
"""
def tfidf_LDA(corpus_matrix):

    perplexity_metrics=[]
    score_metrics=[]
    for i in range(20,110,5):
        lda=LDA(n_components=i, random_state=0, n_jobs=6, learning_decay=0.7)
        clf=lda.fit(corpus_matrix)
        perplexity_metrics.append(clf.perplexity(corpus_matrix))
        score_metrics.append(clf.score(corpus_matrix))
        res=clf.transform(corpus_matrix)
        np.save(str(i)+'transformed.npy',res,allow_pickle=True)

    n_topics=[i for i in range(20,110,5)]

    # Show graph
    plt.figure(figsize=(12, 8))
    plt.plot(n_topics, perplexity_metrics, label='perplexity')
    plt.plot(n_topics, score_metrics, label='log likelihood')
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Log Likelyhood and Perplexity Scores")
    plt.legend(title='Scores', loc='best')
    plt.savefig('choose_best.pdf')
    plt.show()




corpus_matrix=np.load('bigrams_corpus_matrix.npy',allow_pickle=True)
feature_list=np.load('bigrams_wordfeature_list.npy',allow_pickle=True)

print('load successfully')

tfidf_LDA(corpus_matrix)

print('This is the end')





