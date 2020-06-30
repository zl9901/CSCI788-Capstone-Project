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


# clean up data
stemmer = SnowballStemmer('english')
vocab_size = 10000

"""
preprocess the dataset
"""
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
        if token not in STOPWORDS and token not in ['al','cd','et','en','el','da','de','lo','la','le','rt']:
            result.append(lemmatize_stemming(token))
    return result


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
        clf = SVC(kernel='rbf', C=100, gamma='scale',class_weight=cw)

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
            clf = SVC(kernel='rbf', C=100, gamma='scale', class_weight=dic)

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



# corona_body_all_text, corona_abstract_all_text, spreadsheet_match, keywords_list = preprocess_data()
# print('The total number of documents is '+str(len(corona_body_all_text)))
# print('The total number of items in spreadsheet is '+str(len(spreadsheet_match)))
# print('The total number of documents which contain keywords is '+str(len(keywords_list)))



# body_all_text = [preprocess_stem_clean(x) for x in corona_body_all_text]
# print('words cleaning completed')
# corpus_matrix,word_feature_list=tfidf_LDA(body_all_text)


corpus_matrix=np.load('corpus_matrix.npy',allow_pickle=True)
# body_all_text=np.load('body_all_text.npy',allow_pickle=True)
print('load successfully')


# np.save('word_feature_list.npy',word_feature_list,allow_pickle=True)
# np.save('corpus_matrix.npy',corpus_matrix,allow_pickle=True)

print('dimension reduction is done')




image_id=0
training_accuracy,test_accuracy,training_precision,test_precision,training_recall,test_recall,training_f1_score,test_f1_score=add_noise(corpus_matrix)
plot_confidence_intervals(training_accuracy,test_accuracy,image_id,'Accuracy')
image_id+=1
plot_confidence_intervals(training_precision,test_precision,image_id,'Precision')
image_id+=1
plot_confidence_intervals(training_recall,test_recall,image_id,'Recall')
image_id+=1
plot_confidence_intervals(training_f1_score,test_f1_score,image_id,'F1_score')


print('this is the end')



















