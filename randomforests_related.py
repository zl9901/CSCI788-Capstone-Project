
import xlsxwriter
import random
import xlrd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import numpy as np


def randomforests_predict(corpus_matrix,y,indices,spreadsheet_match):

    clf=RandomForestClassifier(n_estimators=300, n_jobs=12, bootstrap=False)
    train_features=[]
    for i in range(len(corpus_matrix)):
        if i in indices:
            # train on 200 papers
            train_features.append(corpus_matrix[i])

    train_features=np.array(train_features)
    y=np.array(y)
    corpus_matrix=np.array(corpus_matrix)

    # predict on the entire dataset
    y_pred=clf.fit(train_features,y).predict(corpus_matrix)


    workbook=xlsxwriter.Workbook('spreadsheet_randomforests.xlsx')
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
use first 2000 papers in the excel file which doesn't include
#201, #202, #203 paper
"""
def test(spreadsheet_match):

    record=[201,202,203]
    y_pred=[0,1,0]
    workbook = xlsxwriter.Workbook('spreadsheet_randomforests.xlsx')
    worksheet = workbook.add_worksheet('My sheet')

    # rewrite a file, this is different from the initial one
    header = ['paper_id', 'title', 'authors', 'paper_number', 'label']
    for k in range(len(header)):
        worksheet.write(0, k, header[k])

    loc = ('labels.xlsx')
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    # return labels and number of the paper
    for i in range(1, sheet.nrows):
        for j in range(len(spreadsheet_match[0])):
            worksheet.write(i, j, sheet.cell_value(i, j))

    for p in range(2001, 2001 + len(record)):
        for q in range(len(spreadsheet_match[0])):
            worksheet.write(p, q, spreadsheet_match[record[p - 2001]][q])
            worksheet.write(p, 4, y_pred[p - 2001])
    workbook.close()



def test_randomforests(corpus_matrix):
    y = []
    indices = []
    loc = ('spreadsheet_randomforests.xlsx')

    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    # return labels and position of the paper
    for i in range(1, sheet.nrows):
        y.append(int(sheet.cell_value(i, 4)))
        indices.append(int(sheet.cell_value(i, 3)))

    # shuffle the entire dataset
    tmp=list(zip(indices,y))
    random.shuffle(tmp)
    indices,y=zip(*tmp)

    # split the data set
    training_features=[]
    training_labels=[]
    testing_features=[]
    testing_labels=[]
    # 70% for training
    for i in range(43000):
        training_features.append(corpus_matrix[indices[i]])
        training_labels.append(y[i])
    # 30% for testing, 62548 papers in total
    for j in range(43000,len(indices)):
        testing_features.append(corpus_matrix[indices[j]])
        testing_labels.append(y[j])

    # use randomforests to make the prediction
    clf=RandomForestClassifier(n_estimators=300,n_jobs=12,bootstrap=False)
    clf.fit(training_features,training_labels)
    y_pred=clf.predict(testing_features)
    print("Accuracy:", accuracy_score(testing_labels, y_pred))