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




from bert_implementation import *
from randomforests_related import *
from svm_related import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

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


def remove_punctuation(string):

    # punctuation marks
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # traverse the given string and if any punctuation
    # marks occur replace it with null
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")

            # Print string without punctuation
    return string.split(' ')


# %% [code]
def preprocess_data():
    i,k = 0,0
    # word related to diagnositics and surveillance
    related_words = ['diagnostic', 'diagnostics', 'symptomatic', 'diagnosing', 'diagnosis', 'clinical',
                     'diagnose', 'diagnoses', 'detection', 'screening', 'analytical', 'assessment',
                    'prognosis', 'surveillance', 'monitoring', 'reconnaissance']
    spreadsheet_match = []
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
            # this variable is used to determine how many sections may exist
            author_cnt=0
            if i % 1000 == 0:
                print ("Working (number %d)..." % i)

            """
            This is for test purpose
            """
            # if i==200:
            #     return corona_body_all_text, corona_abstract_all_text, spreadsheet_match, keywords_list

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
                    abstract_text = ''

                """
                extract all the keywords
                """
                tmp=''
                name = ''
                # d represents each individual dictionary
                # j['body_text'] is a python list which contains many dictionaries
                for d in j['body_text']:
                    # preprocess the data which will bu put in csv file

                    if author_cnt<1:
                        for dic in j['metadata']['authors']:
                            name += dic['first'] + ' ' + dic['last']
                            name += ', '
                        name = name[:-2]
                        author_cnt+=1


                    if d['section']=='Keywords':
                        tmp+=d['text']
                        keywords_list.append(tmp)

                spreadsheet_match.append([j['paper_id'], j['metadata']['title'], name, i, 1])

                i+=1


                """
                body text and abstract consist the whole body text
                """
                body_text = ' '.join(x['text'] for x in j['body_text'])
                body_text += ' ' + abstract_text

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
    return corona_body_all_text, corona_abstract_all_text, spreadsheet_match, keywords_list



"""
we use wordCloud library to show the most frequent words in each cluster 
"""
def word_cloud_advanced(corona_pos_all_text,image_id):

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
    string='wordcloud'+str(image_id)+'.pdf'
    plt.savefig(string)
    plt.show()



"""
dimension reduction
"""
def tfidf_LDA(corona_body_all_text):
    corpus=[]
    for text in corona_body_all_text:
        corpus.append(' '.join(text))

    # the features of each data are 900 now
    vectorizer=TfidfVectorizer(max_features=100, stop_words='english')
    corpus_matrix=vectorizer.fit_transform(corpus)
    print(corpus_matrix.shape)


    # the most representative top 1500 words in corona_body_all_text
    # the basic structure is like a python dictionary
    word_feature_list=vectorizer.get_feature_names()
    corpus_matrix=corpus_matrix.toarray()
    return corpus_matrix,word_feature_list



"""
this is Gaussian mixture models implementation
"""
def Gaussian_mixture_models(corpus_matrix,k):

    gmm=mixture.GaussianMixture(n_components=k)
    y_gmm=gmm.fit_predict(corpus_matrix)

    centers=gmm.means_

    bic_score=gmm.bic(corpus_matrix)
    aic_score=gmm.aic(corpus_matrix)

    return y_gmm, centers, bic_score, aic_score


def GMM_visualization(corpus_matrix):

    corpus_copy=copy.copy(corpus_matrix)
    np.random.shuffle(corpus_copy)
    bayesian_info_criterion = []
    akaike_info_criterion =[]

    n_components=np.arange(2,30)
    for k in range(2, 30):
        indices, centers, bic_score, aic_score=Gaussian_mixture_models(corpus_copy[:1000],k)
        bayesian_info_criterion.append(bic_score)
        akaike_info_criterion.append(aic_score)

    plt.plot(n_components,bayesian_info_criterion,label='BIC')
    plt.plot(n_components,akaike_info_criterion,label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('scores')
    plt.savefig('Bayesian&Akaike_Information_Criterion.pdf')
    plt.show()



def GMM_wordCloud(corona_info, indices,image_id):
    """
    length of res corresponds to number of clusters
    """
    res=[[] for _ in range(len(set(indices)))]
    for i in range(len(indices)):
        res[indices[i]].append(corona_info[i][:])

    """
    for different clusters show different wordCloud
    """
    for subarray in res:
        word_cloud_advanced(subarray,image_id)
        image_id+=1



"""
data preprocessing and wordcloud operation
"""
image_id=0
corona_body_all_text, corona_abstract_all_text, spreadsheet_match, keywords_list = preprocess_data()
print('The total number of documents is '+str(len(corona_body_all_text)))
print('The total number of items in spreadsheet is '+str(len(spreadsheet_match)))
print('The total number of documents which contain keywords is '+str(len(keywords_list)))


keywords_all_text = [preprocess_stem_clean(x) for x in keywords_list]
word_cloud_advanced(keywords_all_text,image_id)
image_id+=1


"""
this body_all_text below is after stem cleaning
"""
# body_all_text = [preprocess_stem_clean(x) for x in corona_body_all_text]


my_body_all_text=np.load('body_all_text.npy',allow_pickle=True)

"""
dimension reduction
"""
corpus_matrix,word_feature_list=tfidf_LDA(my_body_all_text)

np.save('hundred_corpus_matrix.npy',corpus_matrix,allow_pickle=True)
print('saved successfully')

my_corpus_matrix=np.load('hundred_corpus_matrix.npy',allow_pickle=True)
my_word_feature_list=np.load('word_feature_list.npy',allow_pickle=True).tolist()

print(my_corpus_matrix.shape)
print(my_body_all_text.shape)
print(len(my_word_feature_list))



"""
GMM Implementation
"""
GMM_visualization(my_corpus_matrix)
indices,_,_,_=Gaussian_mixture_models(my_corpus_matrix,8)
GMM_wordCloud(my_body_all_text,indices,image_id)


print('This is the end')





