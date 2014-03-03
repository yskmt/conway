import numpy as np
import csv

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
# from sklearn import svm
# from sklearn.linear_model import SGDClassifier
# from sklearn import cross_validation
# from sklearn.neighbors import KNeighborsClassifier

import time
import pdb
import sys
import gc
import os 


def generate_symmetry( data ):
    "generate the symmetry cell pattern"
    # data: n*400 array
    
    n,m = data.shape
    data_symm = np.zeros([5*n, m], dtype=bool)

    for l in range(n):
        
        for k in range(400):
            i = k%20
            j = k/20

            k2 = (19-j) + 20*i
            k3 = (19-i) + 20*(19-j)
            k4 = j      + 20*(19-i)
            k5 = (19-i) + 20*j
            k6 = i      + 20*(19-j)

            data_symm[(l-1)*5,k2]   = data[l,k]
            data_symm[(l-1)*5+1,k3] = data[l,k]
            data_symm[(l-1)*5+2,k4] = data[l,k]
            data_symm[(l-1)*5+3,k5] = data[l,k]
            data_symm[(l-1)*5+4,k6] = data[l,k]

    return data_symm

#######################################################################

## Starting the program

# prepare the directories
if not os.path.exists("outs"):
    os.mkdir("outs")
if not os.path.exists("save"):
    os.mkdir("save")

if len(sys.argv)<5:
    print "not enough arguments!!"
    sys.exit()

arglist = sys.argv

# delta range
min_delta = int (arglist[1])
max_delta = int (arglist[2])
# model range
min_models = int (arglist[3]) # >=0
max_models = int (arglist[4]) # <=400
n_models = max_models-min_models
# neighborhood range
nei_range = 20
# training set size
train_size = 1.0

# read in  data, parse into training and target sets
print "reading the traindata"
traindata = np.loadtxt(open('Data/train.csv','r'), skiprows=1, delimiter=',', dtype='int')
n_train, m_train = traindata.shape

# clean up the traindatas
print "cleaning up the traindata"
id = traindata[:,0]
delta = traindata[:,1]

traindata_st = [[] for i in range(5) ]
traindata_ed = [[] for i in range(5) ]
for i in range(max(delta)):
    # extract rows with each delta value
    if min_delta<=i<max_delta:
        traindata_st[i] = np.array(traindata[traindata[:, 1] == (i+1), 2:402], dtype=int)
        traindata_ed[i] = np.array(traindata[traindata[:, 1] == (i+1), 402:802], dtype=int)

# reshaping the arrays
print "reshaping the arrays..."

n_train = np.zeros(max_delta, dtype=int)
train_st = [[] for i in range(5) ]
train_ed = [[] for i in range(5) ]

for i in range(min_delta, max_delta):
    n_train[i] = int(len(traindata_st[i])*train_size)

    train_st[i] = np.array(traindata_st[i][:n_train[i]], dtype=bool, order='F')
    train_ed[i] = np.array(traindata_ed[i][:n_train[i]], dtype=bool, order='F')

    # add 5 more symmetrical positions when we do full model simulation
    # train_st[i] = \
    #               np.asfortranarray(np.concatenate((train_st[i], generate_symmetry(train_st[i]))))
    # train_ed[i] = \
    #               np.asfortranarray(np.concatenate((train_ed[i], generate_symmetry(train_ed[i]))))
    # n_train[i] = int(len(train_st[i]))

# try to free some memoery
del traindata
del traindata_st
del traindata_ed

# solve the problem for different levels
for i in range(min_delta, max_delta):
    
    # run the svm
    print "runnning the fit for delta = {0:d}, models = {1:d} to {2:d}".format((i+1), min_models, max_models)

    st_clf = time.time()
    clf = RandomForestClassifier(n_estimators=20, n_jobs=1)
    if n_models == 1:
        clf.fit(train_ed[i], train_st[i][:,min_models])
    else:
        clf.fit(train_ed[i], train_st[i][:,min_models:max_models])
    ed_clf = time.time()
    print "fit complete. elapsed time = %f" %(ed_clf-st_clf)
    
    # save the clf
    print "saving the clfs for delta = %i" %(i+1)
    savedir = 'save/cw_'+arglist[1]+"_"+arglist[2]+"_"+arglist[3]+"_"+arglist[4]
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    savefile = savedir+"/"+arglist[1]+"_"+arglist[2]+"_"+arglist[3]+"_"+arglist[4]+".pkl"
    joblib.dump(clf, savefile) 


#### Data Prediction ####

# read in  data, parse into training and target sets
print "reading the testdata"
testdata = np.loadtxt(open('Data/test.csv','r'), skiprows=1, delimiter=',', dtype='int')
n_test,m_test = testdata.shape

# clean up the testdata
print "cleaning up the testdata"
id_test = testdata[:,0]
delta_test = testdata[:,1]

# extract the end data of the testdata
test_st = [[] for i in range(5) ]
test_ed = [[] for i in range(5) ]
for i in range(max(delta)):
    # extract rows with each delta value
    if min_delta<=i<max_delta:
        test_ed[i] = np.array(testdata[testdata[:, 1] == (i+1), 2:402], dtype=int)

# predict the values for different levels
for i in range(min_delta, max_delta):
    print "predicting the values for delta = %i" %(i+1)
    st_prd = time.time()
    test_st[i] = clf.predict(test_ed[i])
    ed_prd = time.time()
    print "prediction complete. elasped time = %f" %(ed_prd-st_prd)

# output results
outfile = 'outs/out_'+arglist[1]+"_"+arglist[2]+"_"+arglist[3]+"_"+arglist[4]+".txt"
for i in range(min_delta, max_delta):
    test_st[i].astype(int).tofile(outfile, ',')


# error calc
# er = np.zeros(max_delta)
# for i in range(min_delta, max_delta):
#     C = test_st[i][:,min_models:max_models]
#     if n_models==1:
#         C = C[:,0]
#         er[i] = float(sum(abs(C.astype(int) - (testdata_st[i]).astype(int)))) / \
#                 float(n_test[i]*n_models)
#     else:
#         er[i] = sum(sum(abs(C.astype(int) - (testdata_st[i]).astype(int)))) / \
#                 (n_test[i]*n_models)

# print er
