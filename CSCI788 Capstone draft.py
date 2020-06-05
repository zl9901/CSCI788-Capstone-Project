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
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import csv
import xlsxwriter
import random
import re

import tensorflow as tf
import tensorflow_hub as hub
import bert
from tensorflow.keras.models import Model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from BERT_Implementation.py import *

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
    i,k = 0,0
    # word related to diagnositics and surveillance
    related_words = ['diagnostic', 'diagnostics', 'symptomatic', 'diagnosing', 'diagnosis', 'clinical',
                     'diagnose', 'diagnoses', 'detection', 'screening', 'analytical', 'assessment',
                    'prognosis', 'surveillance', 'monitoring', 'reconnaissance']
    keywords_match = []
    keywords_no_match = []
    corona_abstract_all_text = []
    corona_body_all_text = []

    keywords_list = []

    """
    walk through all the files under specific directory
    """
    for dirname, _, filenames in os.walk('C:/PythonWorkspace/document_parses/pdf_json'):
        for filename in filenames:
            #print(os.path.join(dirname, filename))

            topic_related = False
            keywords_related = False
            if i % 1000 == 0:
                print ("Working (number %d)..." % i)

            """
            This is for test purpose
            """
            # if i==1000:
            #     return corona_pos_all_text, corona_info, keywords_cnt, keywords_list

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
                i+=1
                # cnt is used to count there exists how many keyword sections
                cnt=0
                # d represents each individual dictionary
                # j['body_text'] is a python list which contains many dictionaries
                for d in j['body_text']:
                    # preprocess the data which will bu put in csv file
                    name = ''
                    for dic in j['metadata']['authors']:
                        name += dic['first'] + ' ' + dic['last']
                        name += ', '
                    name = name[:-2]

                    if d['section']=='Keywords':
                        cnt+=1
                        tmp+=d['text']
                        keywords_list.append(tmp)
                        if cnt==1:
                            keywords_match.append([j['paper_id'], j['metadata']['title'], name, i, 1])
                        keywords_related=True
                if not keywords_related:
                    keywords_no_match.append([j['paper_id'], j['metadata']['title'], name, i, 0])


                """
                body text and abstract consist the whole body text
                """
                body_text = ' '.join(x['text'] for x in j['body_text'])
                body_text += " " + abstract_text

                """
                for related_word in related_words:
                    if related_word in body_text:
                        topic_related = True
                        break
                if topic_related:
                    if abstract_text:
                        corona_pos_all_text.append(abstract_text)
                """

                # append abstract content
                if abstract_text:
                    k+=1
                    corona_abstract_all_text.append(abstract_text)
                # append body text content
                corona_body_all_text.append(body_text)
        print(i)
        print(k)
    return corona_body_all_text, corona_abstract_all_text, keywords_match, keywords_no_match, keywords_list



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
    plt.savefig('wordcloud.pdf')
    plt.show()



def generate_histogram(corona_body_all_text):
    # corona_hist = [len(preprocess_stem_clean(x)) for x in corona_body_all_text]
    corona_hist = [len(re.findall(r'\w+', x))  for x in corona_body_all_text]
    plt.hist(corona_hist,bins='auto',color='#0504aa',alpha=0.7)
    # plt.bar([i for i in range(len(corona_pos_all_text))], corona_pos_all_text)
    plt.xlabel('Number of words')
    plt.ylabel('Number of articles')
    plt.savefig('histogram.pdf')
    plt.show()


def generate_spreadsheet(keywords_match,keywords_no_match):
    match_shuffled = random.sample(keywords_match, len(keywords_match))[:100]
    no_match_shuffled = random.sample(keywords_no_match, len(keywords_no_match))[:100]
    combination_shuffled=match_shuffled+no_match_shuffled

    workbook=xlsxwriter.Workbook('spreadsheet.xlsx')
    worksheet=workbook.add_worksheet('My sheet')

    for i in range(len(combination_shuffled)):
        for j in range(len(combination_shuffled[0])):
            worksheet.write(i,j,combination_shuffled[i][j])
    workbook.close()




def generate_csv(keywords_match,keywords_no_match):
    match_shuffled = random.sample(keywords_match, len(keywords_match))[:100]
    no_match_shuffled = random.sample(keywords_no_match, len(keywords_no_match))[:100]

    file=open('labels.csv','w+',newline='',encoding='utf-8')
    # identifying header
    header=[['paper_id','title','authors','paper_number','keyword_match']]

    # writing data row-wise into the csv file
    with file:
        write=csv.writer(file)
        write.writerows(header)
        write.writerows(match_shuffled)
        write.writerows(no_match_shuffled)





def generate_tfidf(corona_body_all_text):
    corpus=[]
    for text in corona_body_all_text:
        corpus.append(' '.join(text))

    vectorizer=TfidfVectorizer(max_features=5000)
    corpus_matrix=vectorizer.fit_transform(corpus)
    print(corpus_matrix.shape)
    # the most representative top 5000 words in corona_body_all_text
    word_feature_list=vectorizer.get_feature_names()
    corpus_matrix=corpus_matrix.toarray()
    return corpus_matrix,word_feature_list


"""
data preprocessing and wordcloud operation
"""
corona_body_all_text, corona_abstract_all_text, keywords_match, keywords_no_match, keywords_list = preprocess_data()
print('The total number of documents is '+str(len(corona_body_all_text)))
print('The total number of matches is '+str(len(keywords_match)+len(keywords_no_match)))
print('The total number of documents which contain keywords is '+str(len(keywords_list)))
keywords_all_text = [preprocess_stem_clean(x) for x in keywords_list]
word_cloud_advanced(keywords_all_text)

"""
dimension reduction
"""
#generate_histogram(corona_body_all_text)
body_all_text = [preprocess_stem_clean(x) for x in corona_body_all_text]
corpus_matrix,word_feature_list=generate_tfidf(body_all_text)


"""
spreadsheet visualization
"""
generate_csv(keywords_match,keywords_no_match)
generate_spreadsheet(keywords_match,keywords_no_match)


"""
BERT  Algorithm Implementation
"""
# embeddings_info=generate_bert(corona_abstract_all_text)
# print(len(embeddings_info))
# print(len(embeddings_info[0]))
