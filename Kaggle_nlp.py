# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


import os
# import pdb
import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json as js


import nltk
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer,SnowballStemmer
# from nltk.stem.porter import *


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
        if token not in STOPWORDS and token not in ['al','cd','et','en','el','da','de','lo','la','le','rt']:
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

    # the features of each data are 100 now
    vectorizer=TfidfVectorizer(max_features=1500, stop_words='english')
    corpus_matrix=vectorizer.fit_transform(corpus)
    print(corpus_matrix.shape)


    # the most representative top 1500 words in corona_body_all_text
    # the basic structure is like a python dictionary
    word_feature_list=vectorizer.get_feature_names()
    corpus_matrix=corpus_matrix.toarray()
    return corpus_matrix,word_feature_list


"""
return the labels and number/id of specific paper
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


"""
this function is designed to extract the labels of specific papers
"""
def extract_info(filename,tmp):

    indices=[]
    y=[]
    loc = (filename)

    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    # return labels and number of the paper
    for i in range(len(tmp)):
        indices.append(int(sheet.cell_value(tmp[i], 3)))
        y.append(int(sheet.cell_value(tmp[i], 4)))
    return y,indices


"""
estimate the effect of the training dataset size
"""
def add_noise(corpus_matrix):

    cnt=0

    training_accuracy=[[] for _ in range(11)]
    test_accuracy=[[] for _ in range(11)]

    training_precision=[[] for _ in range(11)]
    test_precision=[[] for _ in range(11)]

    training_recall=[[] for _ in range(11)]
    test_recall=[[] for _ in range(11)]

    training_f1_score=[[] for _ in range(11)]
    test_f1_score=[[] for _ in range(11)]


    """
    first 40 development sets all need to be randomly generated and tested their MSE
    """
    for index in range(40):

        # 5 represents 5 different numeric levels
        ini_training_y, ini_training_indices, _ = generate_development_set('labels.xlsx')
        ini_training_y = np.array(ini_training_y)

        cw = collections.Counter(ini_training_y)
        clf = SVC(kernel='linear', C=100, gamma='scale',class_weight=cw)

        ini_training_X = []
        for i in range(len(corpus_matrix)):
            if i in ini_training_indices:
                ini_training_X.append(corpus_matrix[i])
        ini_training_X = np.array(ini_training_X)


        ini_obj=clf.fit(ini_training_X, ini_training_y)
        ini_training_y_pred = ini_obj.predict(ini_training_X)
        ini_training_y_score = ini_obj.decision_function(ini_training_X)

        ini_training_ac = accuracy_score(ini_training_y, ini_training_y_pred)
        ini_training_pr = precision_score(ini_training_y, ini_training_y_pred)
        ini_training_re = recall_score(ini_training_y, ini_training_y_pred)
        ini_training_f1 = f1_score(ini_training_y, ini_training_y_pred)

        training_accuracy[0].append(ini_training_ac)
        training_precision[0].append(ini_training_pr)
        training_recall[0].append(ini_training_re)
        training_f1_score[0].append(ini_training_f1)

        # only plot the last ROC curve of 40 random results
        if index==39:
            ini_training_fpr, ini_training_tpr, threshold = roc_curve(ini_training_y, ini_training_y_score)

            # plot the ROC curve
            plt.figure()
            plt.plot(ini_training_fpr, ini_training_tpr, color='darkorange', lw=2)
            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('The ROC curve')
            string = 'ROC_curve_training' + str(cnt) + '.pdf'
            plt.savefig(string)
            plt.show()



        ini_test_y, ini_test_indices = generate_labels('200_samples.xlsx')
        ini_test_y=np.array(ini_test_y)
        ini_test_X = []
        for j in range(len(corpus_matrix)):
            if j in ini_test_indices:
                ini_test_X.append(corpus_matrix[j])
        ini_test_X = np.array(ini_test_X)


        ini_test_y_pred = ini_obj.predict(ini_test_X)
        ini_test_y_score = ini_obj.decision_function(ini_test_X)

        ini_test_ac = accuracy_score(ini_test_y, ini_test_y_pred)
        ini_test_pr = accuracy_score(ini_test_y, ini_test_y_pred)
        ini_test_re = accuracy_score(ini_test_y, ini_test_y_pred)
        ini_test_f1 = accuracy_score(ini_test_y, ini_test_y_pred)

        test_accuracy[0].append(ini_test_ac)
        test_precision[0].append(ini_test_pr)
        test_recall[0].append(ini_test_re)
        test_f1_score[0].append(ini_test_f1)

        if index == 39:
            ini_test_fpr, ini_test_tpr, threshold = roc_curve(ini_test_y, ini_test_y_score)

            # plot the ROC curve
            plt.figure()
            plt.plot(ini_test_fpr, ini_test_tpr, color='darkorange', lw=2)
            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('The ROC curve')
            string = 'ROC_curve_test' + str(cnt) + '.pdf'
            plt.savefig(string)
            plt.show()

            cnt += 1


    """
    from 200*(2**0) to 200*(2**4) training data size
    """
    for k in range(1,11):

        develop_y, develop_indices, pos=generate_development_set('labels.xlsx')
        """
        the maximal range is the number of items which are in the training dataset
        consider edge case , 0 will not be included, 0 represents the head line
        """
        ref = list(set([i for i in range(1,3001)]) - set(pos))

        # generate 40 different training datasets in order to plot error bars
        for pos in range(40):

            # to avoid raising exceptions, sample size can't exceed the the size of the entire array
            # 10 represents there are 10 sampling processes in total
            threshold = len(ref)//10*k
            if threshold >= len(ref):
                threshold = len(ref)
            random_pos = random.sample(ref, threshold)

            # concatenate development set and sampled set as a new training dataset
            random_y, random_indices = extract_info('labels.xlsx',random_pos)
            training_indices=develop_indices+random_indices
            training_y=develop_y+random_y

            training_y=np.array(training_y)

            dic = collections.Counter(training_y)
            clf = SVC(kernel='linear', C=100, gamma='scale', class_weight=dic)

            training_X = []
            for i in range(len(corpus_matrix)):
                if i in training_indices:
                    training_X.append(corpus_matrix[i])
            training_X=np.array(training_X)

            obj = clf.fit(training_X, training_y)
            training_y_pred = obj.predict(training_X)
            training_y_score = obj.decision_function(training_X)


            training_ac = accuracy_score(training_y, training_y_pred)
            training_pr = precision_score(training_y, training_y_pred)
            training_re = recall_score(training_y, training_y_pred)
            training_f1 = f1_score(training_y, training_y_pred)

            training_accuracy[k].append(training_ac)
            training_precision[k].append(training_pr)
            training_recall[k].append(training_re)
            training_f1_score[k].append(training_f1)

            # only plot the last ROC curve of 40 iterations
            if pos==39:
                training_fpr, training_tpr, threshold = roc_curve(training_y, training_y_score)

                # plot the ROC curve
                plt.figure()
                plt.plot(training_fpr, training_tpr, color='darkorange', lw=2)
                plt.xlim([0.0, 1.05])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('The ROC curve')
                string = 'ROC_curve_training' + str(cnt) + '.pdf'
                plt.savefig(string)
                plt.show()


            test_y, test_indices = generate_labels('200_samples.xlsx')
            test_X = []
            test_y=np.array(test_y)
            for j in range(len(corpus_matrix)):
                if j in test_indices:
                    test_X.append(corpus_matrix[j])
            test_X=np.array(test_X)


            test_y_pred = obj.predict(test_X)
            test_y_score = obj.decision_function(test_X)

            test_ac = accuracy_score(test_y, test_y_pred)
            test_pr = precision_score(test_y, test_y_pred)
            test_re = precision_score(test_y, test_y_pred)
            test_f1 = f1_score(test_y, test_y_pred)

            test_accuracy[k].append(test_ac)
            test_precision[k].append(test_pr)
            test_recall[k].append(test_re)
            test_f1_score[k].append(test_f1)

            if pos==39:
                test_fpr, test_tpr, threshold = roc_curve(test_y, test_y_score)

                # plot the ROC curve
                plt.figure()
                plt.plot(test_fpr, test_tpr, color='darkorange', lw=2)
                plt.xlim([0.0, 1.05])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('The ROC curve')
                string = 'ROC_curve_test' + str(cnt) + '.pdf'
                plt.savefig(string)
                plt.show()

                cnt += 1

    return training_accuracy,test_accuracy,training_precision,test_precision,training_recall,test_recall,training_f1_score,test_f1_score


"""
visualization of the confidence intervals
"""
def plot_confidence_intervals(training_error,test_error,image_id,y_name):
    training_plot=[]
    test_plot=[]

    for i in range(len(training_error)):
        # find the middle value of each error
        train_tmp=sorted(training_error[i])
        training_plot.append(train_tmp[len(train_tmp)//2])

        test_tmp=sorted(test_error[i])
        test_plot.append(test_tmp[len(test_tmp)//2])


    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='Confidence interval curve')
    X=[200+i*280 for i in range(11)]
    plt.plot(X,training_plot,label='Training'+' '+y_name)
    plt.plot(X,test_plot,label='Test'+' '+y_name)

    for i in range(len(training_error)):
        plt.vlines(x=X[i],ymin=min(training_error[i]),ymax=max(training_error[i]),linewidth=4,color='r')
    for j in range(len(test_error)):
        plt.vlines(x=X[j], ymin=min(test_error[j]), ymax=max(test_error[j]), linewidth=4, color='g')

    string='confidence_level'+str(image_id)+'.pdf'
    plt.legend(loc='best')
    plt.xlabel('Number of training examples')
    plt.ylabel(y_name)
    plt.savefig(string)
    plt.show()


"""
this function is designed to generate development set
"""
def generate_development_set(filename):
    indices=[]
    y=[]
    pos=[]
    loc = (filename)
    zero=0
    one=0

    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    """
    the maximal range is the number of items which are in the training dataset
    consider edge case , 0 will not be included, 0 represents the head line
    """
    tmp=[i for i in range(1,3001)]
    tmp_copy = copy.copy(tmp)
    np.random.shuffle(tmp_copy)


    # only need 100 positive labels and 100 negative labels
    for i in range(len(tmp_copy)):
        if zero==100 and one==100:
            break
        if int(sheet.cell_value(tmp_copy[i], 4))==0 and zero!=100:
            zero+=1
            indices.append(int(sheet.cell_value(tmp_copy[i], 3)))
            y.append(int(sheet.cell_value(tmp_copy[i], 4)))
            pos.append(tmp_copy[i])
        elif int(sheet.cell_value(tmp_copy[i], 4))==1 and one!=100:
            one+=1
            indices.append(int(sheet.cell_value(tmp_copy[i], 3)))
            y.append(int(sheet.cell_value(tmp_copy[i], 4)))
            pos.append(tmp_copy[i])
    return y,indices,pos

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

    n_components=np.arange(2,31)
    for k in range(2,31):
        # shuffled 10000 papers
        indices, centers, bic_score, aic_score=Gaussian_mixture_models(corpus_copy[:10000],k)
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
        # top 3000 words of each paper can represent the main idea of the paper
        res[indices[i]].append(corona_info[i][:3000])

    """
    for different clusters show different wordCloud
    """
    for subarray in res:
        word_cloud_advanced(subarray,image_id)
        image_id+=1

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


"""
dimension reduction
"""
def LDA_modeling(corona_body_all_text):
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
dimension reduction
"""
def LDA_transformation(corpus_matrix):

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
# np.save('body_all_text.npy',body_all_text,allow_pickle=True)
my_body_all_text=np.load('body_all_text.npy',allow_pickle=True)




"""
dimension reduction
"""
corpus_matrix,word_feature_list=tfidf_LDA(my_body_all_text)
np.save('word_feature_list.npy',word_feature_list,allow_pickle=True)
np.save('corpus_matrix.npy', corpus_matrix, allow_pickle=True)
print('saved successfully')


my_corpus_matrix=np.load('corpus_matrix.npy',allow_pickle=True)
my_word_feature_list=np.load('word_feature_list.npy',allow_pickle=True).tolist()
print('loaded successfully')

print(my_corpus_matrix.shape)
print(my_body_all_text.shape)
print(len(my_word_feature_list))



"""
GMM Implementation
"""
GMM_visualization(my_corpus_matrix)
indices,_,_,_=Gaussian_mixture_models(my_corpus_matrix,15)
GMM_wordCloud(my_body_all_text,indices,image_id)


"""
this is for generating confidence intervals
"""
training_accuracy,test_accuracy,training_precision,test_precision,training_recall,test_recall,training_f1_score,test_f1_score=add_noise(corpus_matrix)
plot_confidence_intervals(training_accuracy,test_accuracy,image_id,'Accuracy')
image_id+=1
plot_confidence_intervals(training_precision,test_precision,image_id,'Precision')
image_id+=1
plot_confidence_intervals(training_recall,test_recall,image_id,'Recall')
image_id+=1
plot_confidence_intervals(training_f1_score,test_f1_score,image_id,'F1_score')

"""
generate histograms
"""
generate_histogram(corona_body_all_text)


"""
run LDA algorithm
"""
LDA_modeling(corona_body_all_text)


"""
run LDA transformation algorithm
"""
LDA_transformation(corpus_matrix)


"""
predict entire dataset
"""
test_performance(corpus_matrix)



print('This is the end')





