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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import mixture



import tensorflow as tf
import tensorflow_hub as hub
import bert
from tensorflow.keras.models import Model


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
            if i==200:
                return corona_body_all_text, corona_abstract_all_text, spreadsheet_match, keywords_list

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
this function is to generate the histogram
"""
def generate_histogram(corona_body_all_text):
    # corona_hist = [len(preprocess_stem_clean(x)) for x in corona_body_all_text]
    # remove all the punctuations at the very beginning
    corona_hist = [len(re.findall(r'\w+', x))  for x in corona_body_all_text]
    plt.hist(corona_hist,bins='auto',color='#0504aa',alpha=0.7)
    # plt.bar([i for i in range(len(corona_pos_all_text))], corona_pos_all_text)
    plt.xlabel('Number of words')
    plt.ylabel('Number of articles')
    plt.savefig('histogram.pdf')
    plt.show()



def initial_spreadsheet(spreadsheet_match):
    # match_shuffled = random.sample(spreadsheet_match, len(spreadsheet_match))[:100]

    workbook=xlsxwriter.Workbook('spreadsheet.xlsx')
    worksheet=workbook.add_worksheet('My sheet')

    header=['paper_id','title','authors','paper_number','label']
    for k in range(len(header)):
        worksheet.write(0,k,header[k])

    for i in range(1,201):
        for j in range(len(spreadsheet_match[0])):
            worksheet.write(i,j,spreadsheet_match[i-1][j])
    workbook.close()




def initial_csv(spreadsheet_match):

    # no_match_shuffled = random.sample(keywords_no_match, len(keywords_no_match))[:100]
    spreadsheet_match=spreadsheet_match[:200]

    file=open('labels.csv','w+',newline='',encoding='utf-8')
    # identifying header
    header=[['paper_id','title','authors','paper_number','label']]

    # writing data row-wise into the csv file
    with file:
        write=csv.writer(file)
        write.writerows(header)
        write.writerows(spreadsheet_match)



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


    """
    LDA (Latent Dirichlet Allocation) Implementation
    -----------------------------------------------------------------------------------------------------------
    """
    # the number of terms included in the bag of words matrix is restricted to the top 1000


    no_features=1500
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(corpus)
    tf_feature_names = tf_vectorizer.get_feature_names()

    """
    this is for the grid search
    """
    search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
    lda = LDA()
    model= GridSearchCV(lda, param_grid=search_params)
    model.fit(tf)

    best_lda_model=model.best_estimator_

    # Model Parameters
    print("Best Model's Params: ", model.best_params_)

    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)

    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(tf))

    # Get Log Likelyhoods from Grid Search Output
    n_topics = [10, 15, 20, 25, 30]
    # print(model.cv_results_)
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
    plt.show()

    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(tf)

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
    df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
    df_document_topics.to_excel("df_document_topics.xlsx")
    print(df_document_topics)

    # Review topics distribution across documents
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    df_topic_distribution.to_excel("df_topic_distribution.xlsx")
    print(df_topic_distribution)


    panel = pyLDAvis.sklearn.prepare(best_lda_model, tf, tf_vectorizer, mds='tsne')
    pyLDAvis.save_html(panel,'LDA_Visualization.html')


    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(best_lda_model.components_)

    # Assign Column and Index
    df_topic_keywords.columns = vectorizer.get_feature_names()
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

    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)

    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
    df_topic_keywords.to_excel("df_topic_keywords.xlsx")
    print(df_topic_keywords)


    """
    grid search ends
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

    """
    -----------------------------------------------------------------------------------------------------------
    """
    corpus_matrix=corpus_matrix.toarray()
    return corpus_matrix,word_feature_list


"""
return the labels and number of specific paper
"""
def generate_labels():
    y=[]
    indices=[]
    loc = ('labels.xlsx')

    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    # return labels and number of the paper
    for i in range(1,sheet.nrows):
        y.append(int(sheet.cell_value(i, 4)))
        indices.append(int(sheet.cell_value(i,3)))
    return y,indices


"""
grid search function
"""
def grid_search(corpus_matrix,y,indices,spreadsheet_match):
    # this is for unbalanced problems
    cw=collections.Counter(y)

    # examples of grid search
    parameters = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 1e-2, 1e-3, 1e-4], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'class_weight':[cw]},
                  {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'class_weight':[cw]}]

    grid = GridSearchCV(SVC(), parameters, refit=True, verbose=3, scoring='f1')

    X = []
    for i in range(len(corpus_matrix)):
        if i in indices:
            # train on 200 papers
            X.append(corpus_matrix[i])

    X=np.array(X)
    y=np.array(y)
    corpus_matrix=np.array(corpus_matrix)


    clf = grid.fit(X, y)

    # print(clf.best_params_)
    # print(clf.best_estimator_)
    # print()

    # predict on the entire dataset
    y_score_rbf = clf.decision_function(corpus_matrix)

    # this yields the minimum and maximum value of svm regressor output
    print(str(min(y_score_rbf)) + '   ' + str(max(y_score_rbf)))
    print()

    """
    this is the test area
    """
    y_pred=clf.predict(corpus_matrix)
    workbook=xlsxwriter.Workbook('spreadsheet_SVM_prediction.xlsx')
    worksheet=workbook.add_worksheet('My sheet')

    # rewrite features and labels to a file, this is different from the initial file
    # write the header first
    header=['paper_id','title','authors','paper_number','label']
    for k in range(len(header)):
        worksheet.write(0,k,header[k])

    for i in range(1, len(corpus_matrix) + 1):
        for j in range(len(spreadsheet_match[0])):
            worksheet.write(i, j, spreadsheet_match[i-1][j])
            worksheet.write(i, 4, y_pred[i-1])
    workbook.close()
    """
    this is the test area
    """


    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    # this is for f1_score, recall, accuracy and ROC curve
    training_score = clf.decision_function(X)
    training_score_normalization = normalization(training_score)
    training_label = clf.predict(X)
    print('accuracy ' + str(accuracy_score(y, training_label)))
    print('f1_score ' + str(f1_score(y, training_label)))
    print('precision_score '+str(precision_score(y,training_label)))
    print('recall_score ' + str(recall_score(y, training_label)))

    fpr, tpr, threshold = roc_curve(y, training_score_normalization)

    # plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('The ROC curve')
    plt.savefig('ROC_curve.pdf')
    plt.show()

    # return index of a sorted list
    sorted_indices = sorted(range(len(y_score_rbf)), key=lambda k: y_score_rbf[k])

    # get the paper numbers which are not in the training set
    subtract_indices = []
    for val in sorted_indices:
        if val not in indices:
            subtract_indices.append(val)

    # get top 10%, bottom 10% and middle 10%
    res = []
    res += subtract_indices[:67]
    res += subtract_indices[-67:]
    res += subtract_indices[len(sorted_indices) // 2:len(sorted_indices) // 2 + 33]
    res += subtract_indices[len(sorted_indices) // 2 - 33:len(sorted_indices) // 2]

    workbook = xlsxwriter.Workbook('spreadsheet_svm.xlsx')
    worksheet = workbook.add_worksheet('My sheet')

    # write to a file without labels, this is different from the initial file
    # each paper is labeled 1 automatically at the very beginning
    # then we need to label each paper manually
    header = ['paper_id', 'title', 'authors', 'paper_number', 'label']
    for k in range(len(header)):
        worksheet.write(0, k, header[k])

    for i in range(1, len(res) + 1):
        for j in range(len(spreadsheet_match[0])):
            worksheet.write(i, j, spreadsheet_match[res[i - 1]][j])
    workbook.close()


def LDA_analysis(body_all_text_processed):
    # Load the library with the CountVectorizer method
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

    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english')
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(body_all_text_processed)

    # Visualise the 10 most common words
    plot_10_most_common_words(count_data, count_vectorizer)


    # This is another version of LDA implementation
    """
    import warnings
    warnings.simplefilter("ignore", DeprecationWarning)

    # Helper function
    def print_topics(model, count_vectorizer, n_top_words):
        words = count_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

    # Tweak the two parameters below
    number_topics = 5
    number_words = 10
    # Create and fit the LDA model, n_jobs=-1 means using all processors
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)
    # Print the topics found by the LDA model
    print("Topics found via LDA:")
    print_topics(lda, count_vectorizer, number_words)
    """

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
Only keep the words with the top-5000 tfidf words.
"""
def process_word(text, word_feature_list):
    result = []
    for p in text:
        temp = []
        for w in p:
            if w in word_feature_list:
                temp.append(w)
        if len(temp) != 0:
            result.append(temp)
    result=np.array(result)
    return result



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
get histogram visualization
"""
# generate_histogram(corona_body_all_text)


"""
this body_all_text below is after stem cleaning
"""
# body_all_text = [preprocess_stem_clean(x) for x in corona_body_all_text]
"""
this body_all_text is for entire words
"""
tmp_list=corona_body_all_text.copy()
body_all_text = [remove_punctuation(x) for x in tmp_list]

"""
dimension reduction
"""
corpus_matrix,word_feature_list=tfidf_LDA(body_all_text)

"""
GMM Implementation
"""
GMM_visualization(corpus_matrix)
body_all_text_processed=process_word(body_all_text,word_feature_list)
print('word screening is done')
indices,_,_,_=Gaussian_mixture_models(corpus_matrix,8)
GMM_wordCloud(body_all_text_processed,indices,image_id)


"""
LDA Implementation
"""
LDA_analysis(corona_body_all_text)
print('dimensional reduction is done')
print()


"""
initial spreadsheet visualization
"""
# initial_csv(spreadsheet_match)
# initial_spreadsheet(spreadsheet_match)


"""
SVM iterations to generate the pre-trained model
this also includes hyperparameter tuning models 
"""
y,indices=generate_labels()
# generate_SVM(corpus_matrix,y,indices,spreadsheet_match)
grid_search(corpus_matrix,y,indices,spreadsheet_match)
# k_fold_svm(corpus_matrix,y,indices,spreadsheet_match)



"""
randomforests to test the pre-trained model
"""
# randomforests_predict(corpus_matrix,y,indices,spreadsheet_match)
# test_randomforests(corpus_matrix)



"""
BERT  Algorithm Implementation
"""
# embeddings_info=generate_bert(corona_abstract_all_text)
# print(len(embeddings_info))
# print(len(embeddings_info[0]))
