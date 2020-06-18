
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.model_selection import KFold
import collections


"""
k-fold analysis function
"""
def k_fold_svm(corpus_matrix,y,indices,spreadsheet_match):
    # this is for unbalanced problems
    cw = collections.Counter(y)

    # initialize an SVM rbf kernel
    svc_rbf = SVC(kernel='rbf', gamma='scale', class_weight=cw)
    X = []

    for i in range(len(corpus_matrix)):
        if i in indices:
            # train on 200 papers
            X.append(corpus_matrix[i])

    X=np.array(X)
    y=np.array(y)
    corpus_matrix=np.array(corpus_matrix)

    scores=[]
    cv=KFold(n_splits=10,random_state=32,shuffle=True)
    store_X=[]
    store_y=[]
    for train_index,test_index in cv.split(X):
        X_train,X_test,y_train,y_test=X[train_index],X[test_index],y[train_index],y[test_index]
        svc_rbf.fit(X_train,y_train)
        scores.append(svc_rbf.score(X_test,y_test))
        store_X.append(X_train)
        store_y.append(y_train)

    pos=scores.index(max(scores))
    # X=store_X[pos].tolist()
    # y=store_y[pos].tolist()
    X=store_X[pos]
    y=store_y[pos]

    clf = svc_rbf.fit(X, y)

    # predict on the entire dataset
    y_score_rbf = clf.decision_function(corpus_matrix)

    # this yields the minimum and maximum value of svm regressor output
    print(str(min(y_score_rbf)) + '   ' + str(max(y_score_rbf)))
    print()

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    # this is for f1_score, recall, accuracy and ROC curve
    training_score = clf.decision_function(X)
    training_score_normalization = normalization(training_score)
    training_label = clf.predict(X)
    print('accuracy ' + str(accuracy_score(y, training_label)))
    print('f1_score ' + str(f1_score(y, training_label)))
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



"""
go through svm iteration
"""
def generate_SVM(corpus_matrix,y,indices,spreadsheet_match):

    # this is for unbalanced problems
    cw=collections.Counter(y)

    # initialize an SVM rbf kernel
    svc_rbf=SVC(kernel='rbf',gamma='scale',class_weight=cw)
    X=[]

    for i in range(len(corpus_matrix)):
        if i in indices:
            # train on 200 papers
            X.append(corpus_matrix[i])

    X=np.array(X)
    y=np.array(y)
    corpus_matrix=np.array(corpus_matrix)

    clf=svc_rbf.fit(X,y)

    # predict on the entire dataset
    y_score_rbf=clf.decision_function(corpus_matrix)

    # this yields the minimum and maximum value of svm regressor output
    print(str(min(y_score_rbf))+'   '+str(max(y_score_rbf)))
    print()

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    # this is for f1_score, recall, accuracy and ROC curve
    training_score=clf.decision_function(X)
    training_score_normalization = normalization(training_score)
    training_label=clf.predict(X)
    print('accuracy ' + str(accuracy_score(np.array(y), training_label)))
    print('f1_score ' + str(f1_score(np.array(y), training_label)))
    print('recall_score ' + str(recall_score(np.array(y), training_label)))

    fpr, tpr, threshold = roc_curve(np.array(y), training_score_normalization)

    # plot the ROC curve
    plt.figure()
    plt.plot(fpr,tpr, color='darkorange', lw=2)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('The ROC curve')
    plt.show()

    # return index of a sorted list
    sorted_indices=sorted(range(len(y_score_rbf)),key=lambda k:y_score_rbf[k])


    # get the paper numbers which are not in the training set
    subtract_indices=[]
    for val in sorted_indices:
        if val not in indices:
            subtract_indices.append(val)


    # get top 10%, bottom 10% and middle 10%
    res=[]
    res+=subtract_indices[:67]
    res+=subtract_indices[-67:]
    res+=subtract_indices[len(sorted_indices)//2:len(sorted_indices)//2+33]
    res+=subtract_indices[len(sorted_indices)//2-33:len(sorted_indices)//2]


    workbook=xlsxwriter.Workbook('spreadsheet_svm.xlsx')
    worksheet=workbook.add_worksheet('My sheet')


    # write to a file without labels, this is different from the initial file
    # each paper is labeled 1 automatically at the very beginning
    # then we need to label each paper manually
    header=['paper_id','title','authors','paper_number','label']
    for k in range(len(header)):
        worksheet.write(0,k,header[k])

    for i in range(1,len(res)+1):
        for j in range(len(spreadsheet_match[0])):
            worksheet.write(i,j,spreadsheet_match[res[i-1]][j])
    workbook.close()