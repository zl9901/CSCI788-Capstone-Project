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
import pyLDAvis
import pyLDAvis.sklearn
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
dimension reduction
"""
def tfidf_LDA(corona_body_all_text):
    corpus=[]
    for text in corona_body_all_text:
        corpus.append(' '.join(text))

    # the number of terms included in the bag of words matrix is restricted to the top 100
    no_features=100
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    corpus_matrix = tf_vectorizer.fit_transform(corpus)
    print(corpus_matrix.shape)

    # the most representative top 1500 words in corona_body_all_text
    # the basic structure is like a python dictionary
    word_feature_list = tf_vectorizer.get_feature_names()


    """
    this is for the grid search
    """
    search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
    lda = LDA()
    model= GridSearchCV(lda, param_grid=search_params, n_jobs=6)
    model.fit(corpus_matrix)

    best_lda_model=model.best_estimator_

    # Model Parameters
    print("Best Model's Params: ", model.best_params_)

    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)

    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(corpus_matrix))

    # Get Log Likelyhoods from Grid Search Output
    n_topics = [10, 15, 20, 25, 30]
    log_likelyhoods_5 = [round(model.cv_results_['mean_test_score'][index]) for index, gscore in enumerate(model.cv_results_['param_learning_decay']) if
                         gscore == 0.5]
    log_likelyhoods_7 = [round(model.cv_results_['mean_test_score'][index]) for index, gscore in enumerate(model.cv_results_['param_learning_decay']) if
                         gscore== 0.7]
    log_likelyhoods_9 = [round(model.cv_results_['mean_test_score'][index]) for index, gscore in enumerate(model.cv_results_['param_learning_decay']) if
                         gscore== 0.9]

    # Show graph
    plt.figure(figsize=(12, 8))
    plt.plot(n_topics, log_likelyhoods_5, label='0.5')
    plt.plot(n_topics, log_likelyhoods_7, label='0.7')
    plt.plot(n_topics, log_likelyhoods_9, label='0.9')
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Log Likelyhood Scores")
    plt.legend(title='Learning decay', loc='best')
    plt.savefig('learning_decay.pdf')
    plt.show()

    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(corpus_matrix)

    # column names
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(len(corona_body_all_text))]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    # Styling
    def color_green(val):
        color = 'green' if val > .1 else 'black'
        return 'color: {col}'.format(col=color)

    def make_bold(val):
        weight = 700 if val > .1 else 400
        return 'font-weight: {weight}'.format(weight=weight)

    # Apply Style
    df_document_topics = df_document_topic.head(len(corona_body_all_text)).style.applymap(color_green).applymap(make_bold)
    df_document_topics.to_excel("df_document_topics.xlsx")
    print(df_document_topics)

    # Review topics distribution across documents
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    df_topic_distribution.to_excel("df_topic_distribution.xlsx")
    print(df_topic_distribution)


    panel = pyLDAvis.sklearn.prepare(best_lda_model, corpus_matrix, tf_vectorizer, mds='tsne')
    pyLDAvis.save_html(panel,'LDA_Visualization.html')


    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(best_lda_model.components_)

    # Assign Column and Index
    df_topic_keywords.columns = tf_vectorizer.get_feature_names()
    df_topic_keywords.index = topicnames

    # View
    df_topic_keywords.head()

    # Show top n keywords for each topic
    def show_topics(vectorizer, lda_model, n_words):
        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    topic_keywords = show_topics(vectorizer=tf_vectorizer, lda_model=best_lda_model, n_words=15)

    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
    df_topic_keywords.to_excel("df_topic_keywords.xlsx")
    print(df_topic_keywords)


    """
    this is for printing the content of all the topics
    """
    # no_topics = 20
    # lda = LDA(n_components=no_topics, learning_method='online', learning_offset=50.,
    #                                random_state=0).fit(tf)
    #
    #
    # def display_topics(model, feature_names, no_top_words):
    #
    #     for topic_idx, topic in enumerate(model.components_):
    #         print("\nTopic #%d:" % topic_idx)
    #         print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    # print()
    # no_top_words = 10
    # display_topics(lda, tf_feature_names, no_top_words)


    sns.set_style('whitegrid')

    # Helper function
    def plot_10_most_common_words(count_data, count_vectorizer):
        words = count_vectorizer.get_feature_names()
        total_counts = np.zeros(len(words))
        # count_data is like a Counter dictionary
        for t in count_data:
            total_counts += t.toarray()[0]

        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))

        plt.figure(2, figsize=(15, 15 / 1.6180))
        plt.subplot(title='10 most common words')
        sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
        sns.barplot(x_pos, counts, palette='husl')
        plt.xticks(x_pos, words, rotation=90)
        plt.xlabel('words')
        plt.ylabel('counts')
        plt.savefig('10_most_common_words.pdf')
        plt.show()


    # Visualise the 10 most common words
    plot_10_most_common_words(corpus_matrix, tf_vectorizer)


    corpus_matrix=corpus_matrix.toarray()
    return corpus_matrix,word_feature_list



"""
data preprocessing and wordcloud operation
"""
corona_body_all_text, corona_abstract_all_text, spreadsheet_match, keywords_list = preprocess_data()
print('The total number of documents is '+str(len(corona_body_all_text)))
print('The total number of items in spreadsheet is '+str(len(spreadsheet_match)))
print('The total number of documents which contain keywords is '+str(len(keywords_list)))



"""
this body_all_text below is after stem cleaning
"""
body_all_text = [preprocess_stem_clean(x) for x in corona_body_all_text]


"""
dimension reduction
"""
corpus_matrix,word_feature_list=tfidf_LDA(body_all_text)


print('This is the end')





