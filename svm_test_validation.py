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


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


import matplotlib.pyplot as plt
import xlrd
import xlsxwriter
import pickle


import random
import copy
import collections
import re
from operator import itemgetter
import seaborn as sns


from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn import mixture

import tensorflow as tf
import tensorflow_hub as hub
import bert
from tensorflow.keras.models import Model

# %% [code]
# clean up data
stemmer = SnowballStemmer('english')
vocab_size = 10000

def preprocess_data():
    i,k = 0,0

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


def random_200_samples(spreadsheet_match):
    index=[]
    loc = ('labels.xlsx')

    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    # return labels and number of the paper
    for i in range(1,sheet.nrows):
        index.append(int(sheet.cell_value(i, 3)))

    ref = list(set([i for i in range(62548)]) - set(index))
    random_sample = random.sample(ref, 200)

    workbook = xlsxwriter.Workbook('200_samples.xlsx')
    worksheet = workbook.add_worksheet('My sheet')

    # write to a file without labels, this is different from the initial file
    # each paper is labeled 1 automatically at the very beginning
    # then we need to label each paper manually
    header = ['paper_id', 'title', 'authors', 'paper_number', 'label']
    for k in range(len(header)):
        worksheet.write(0, k, header[k])

    for i in range(1, len(random_sample) + 1):
        for j in range(len(spreadsheet_match[0])):
            worksheet.write(i, j, spreadsheet_match[random_sample[i - 1]][j])
    workbook.close()
    return random_sample



"""
dimension reduction
"""
def tfidf_LDA(corona_body_all_text):
    corpus=[]
    for text in corona_body_all_text:
        corpus.append(' '.join(text))

    # the features of each data are 900 now
    vectorizer=TfidfVectorizer(max_features=1500, stop_words='english')
    corpus_matrix=vectorizer.fit_transform(corpus)
    print(corpus_matrix.shape)

    # the most representative top 1500 words in corona_body_all_text
    # the basic structure is like a python dictionary
    word_feature_list=vectorizer.get_feature_names()


    corpus_matrix=corpus_matrix.toarray()
    return corpus_matrix,word_feature_list




"""
return the labels and number of specific paper
"""
def generate_labels(filename):
    y=[]
    indices=[]
    loc = (filename)

    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    # return labels and number of the paper
    for i in range(1,sheet.nrows):
        y.append(int(sheet.cell_value(i, 4)))
        indices.append(int(sheet.cell_value(i,3)))
    return y,indices



def test_performance(corpus_matrix):
    training_y,training_indices=generate_labels('labels.xlsx')
    test_y,test_indices=generate_labels('200_samples.xlsx')

    cw = collections.Counter(training_y)

    # examples of grid search
    parameters = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 1e-2, 1e-3, 1e-4], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                   'class_weight': [cw]},
                  {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'class_weight': [cw]}]

    grid = GridSearchCV(SVC(), parameters, refit=True, verbose=3, scoring='f1',n_jobs=6)

    training_X = []
    for i in range(len(corpus_matrix)):
        if i in training_indices:
            # train on 200 papers
            training_X.append(corpus_matrix[i])

    test_X=[]
    for j in range(len(corpus_matrix)):
        if j in test_indices:
            test_X.append(corpus_matrix[j])



    training_X = np.array(training_X)
    training_y = np.array(training_y)
    test_X = np.array(test_X)
    clf = grid.fit(training_X, training_y)


    y_score=clf.decision_function(test_X)
    y_pred=clf.predict(test_X)

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    # this is for f1_score, recall, accuracy and ROC curve
    y_score_normalization = normalization(y_score)
    print('accuracy ' + str(accuracy_score(test_y, y_pred)))
    print('f1_score ' + str(f1_score(test_y, y_pred)))
    print('precision_score '+str(precision_score(test_y,y_pred)))
    print('recall_score ' + str(recall_score(test_y, y_pred)))

    fpr, tpr, threshold = roc_curve(test_y, y_score_normalization)

    # plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('The ROC curve')
    plt.savefig('ROC_curve_test_dataset.pdf')
    plt.show()




corona_body_all_text, corona_abstract_all_text, spreadsheet_match, keywords_list = preprocess_data()
print('The total number of documents is '+str(len(corona_body_all_text)))
print('The total number of items in spreadsheet is '+str(len(spreadsheet_match)))
print('The total number of documents which contain keywords is '+str(len(keywords_list)))


"""
this only needs to be done once
"""
# print(random_200_samples(spreadsheet_match))


body_all_text = [preprocess_stem_clean(x) for x in corona_body_all_text]
print('words cleaning completed')
corpus_matrix,word_feature_list=tfidf_LDA(body_all_text)


np.save('corpus_matrix.npy',corpus_matrix,allow_pickle=True)
np.save('word_feature_list.npy',word_feature_list,allow_pickle=True)
np.save('body_all_text.npy',body_all_text,allow_pickle=True)
print('saved successfully')
my_corpus_matrix=np.load('corpus_matrix.npy',allow_pickle=True)
my_word_feature_list=np.load('word_feature_list.npy',allow_pickle=True).tolist()
my_body_all_text=np.load('body_all_text.npy',allow_pickle=True)
print(my_corpus_matrix.shape)
print(my_body_all_text.shape)
print(len(my_word_feature_list))


test_performance(my_corpus_matrix)


print('this is the end')



















