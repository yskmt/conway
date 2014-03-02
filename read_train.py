import numpy as np
import csv
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation
import time
from sklearn.externals import joblib
import pdb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import sys
import gc
from sklearn import tree

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


def get_delta_neighbor( cell_num, delta ):
    "get the list of delta-neighboring cells"

    i_cell = cell_num%20
    j_cell = cell_num/20

    neighbor_indices = []

    nei_range = delta*2+1

    for i in range(nei_range):
        for j in range(nei_range):
            i_nei = i_cell-delta+i
            j_nei = j_cell-delta+j
            
            if (-1<i_nei<20) & (-1<j_nei<20):
                neighbor_indices.append(i_nei*20+j_nei)
    
    return neighbor_indices


def svm_vectors ( train, target, n_cells, min_models, max_models, delta ):
    "Get the SVM vectors for each cell"
    clfs = []
    start = time.time()
    for i in range(n_cells):
        print "%dth iteration" %i
        st_svc = time.time()
        # clfs.append(SGDClassifier(loss="hinge", penalty="l2"))
        # clfs.append(svm.SVC())
        # clfs.append(KNeighborsClassifier())
        # clfs.append(AdaBoostClassifier(n_estimators=100))
        # clfs.append(RandomForestClassifier(n_estimators=100, n_jobs=4))
        clfs[i].fit(train[:,get_delta_neighbor(i+min_models,delta)], \
                    target[:,i+min_models])

        # pdb.set_trace()
        ed_svc = time.time()
        print "done, time: %f" %(ed_svc-st_svc)        
    end = time.time()

    print "ellapsed time = ", end - start

    return clfs

def svm_predict (clfs, data_test, n_cells, min_models, max_models, delta ):
    "Predict the values for each cell"
    n_train, _ = data_test.shape
    data_predict = np.zeros([n_cells, n_train])

    for i in range(n_cells):
        print "%dth iteration" %i
        st_svc = time.time()
        data_predict[i] = \
            np.array(clfs[i].predict(data_test[:,get_delta_neighbor(i+min_models,delta)]) )
        ed_svc = time.time()
        print "done, time: %f" %(ed_svc-st_svc)        

    return data_predict.T


#######################################################################

## Starting the program

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
train_size = 0.9

# read in  data, parse into training and target sets
print "reading the dataset"
dataset = np.loadtxt(open('Data/train.csv','r'), skiprows=1, delimiter=',', dtype='int')
n,m = dataset.shape

# clean up the datasets
print "cleaning up the dataset"
id = dataset[:,0]
delta = dataset[:,1]

data_st = [[] for i in range(5) ]
data_ed = [[] for i in range(5) ]
for i in range(max(delta)):
    # extract rows with each delta value
    if min_delta<=i<max_delta:
        data_st[i] = np.array(dataset[dataset[:, 1] == (i+1), 2:402], dtype=int)
        data_ed[i] = np.array(dataset[dataset[:, 1] == (i+1), 402:802], dtype=int)

# reshaping the arrays
print "reshaping the arrays..."
                
n_train = np.zeros(max_delta, dtype=int)
n_test = np.zeros(max_delta, dtype=int)
train_st = [[] for i in range(5) ]
train_ed = [[] for i in range(5) ]
test_st = [[] for i in range(5) ]
test_ed = [[] for i in range(5) ]

for i in range(min_delta, max_delta):
    n_train[i] = int(len(data_st[i])*train_size)
    n_test[i] = int(len(data_st[i]) - n_train[i])

    train_st[i] = np.array(data_st[i][:n_train[i]], dtype=bool, order='F')
    train_ed[i] = np.array(data_ed[i][:n_train[i]], dtype=bool, order='F')
    test_st[i] = np.array(data_st[i][n_train[i]:], dtype=bool, order='F')
    test_ed[i] = np.array(data_ed[i][n_train[i]:], dtype=bool, order='F')

    # add 5 more symmetrical positions when we do full model simulation
    train_st[i] = \
        np.array(np.concatenate((train_st[i], generate_symmetry(train_st[i]))),\
                 dtype=bool, order='F')
    train_ed[i] = \
        np.array(np.concatenate((train_ed[i], generate_symmetry(train_ed[i]))),\
                dtype=bool, order='F')

    n_train[i] = int(len(train_st[i]))



# try to free some memoery
dataset = []
data_st = []
data_ed = []
gc.collect()


# classification functions for [delta][cell#]
clfs = [[] for i in range(max_delta)]
# solve the problem for different levels
for i in range(min_delta, max_delta):

    # run the svm
    print "runnning the svm for delta = %i" %(i+1)
    clf = RandomForestClassifier(n_estimators=10, n_jobs=4)

    # clf = tree.DecisionTreeClassifier()
    clf.fit(train_ed[i], train_st[i])


     # clfs[i] =  svm_vectors( train_ed[i], train_st[i], n_models, \
     #                        min_models, max_models, i+nei_range)

    # save the clfs
    # print "saving the clfs for delta = %i" %(i+1)
    # joblib.dump(clfs, 'conway_%i.pkl'%i) 

# predicted data for [delta]
data_predict = [[] for i in range(max_delta)]
# predict the values for different levels
for i in range(min_delta, max_delta):
    print "predicting the values for delta = %i" %(i+1)
    # data_predict[i] = svm_predict( clfs[i], test_ed[i], n_models, \
                                   # min_models, max_models, i+nei_range)
    data_predict[i] = clf.predict(test_ed[i])

er = np.zeros(max_delta)
for i in range(min_delta, max_delta):
    er[i] = sum(sum(abs(test_st[i][:,min_models:max_models].astype(int) - data_predict[i]).T)) / \
        (n_test[i]*n_models)

print er


# output results
outfile = 'outs/out_'+arglist[1]+"_"+arglist[2]+"_"+arglist[3]+"_"+arglist[4]+".txt"
for i in range(min_delta, max_delta):
	data_predict[i].tofile(outfile, ',')


# with open('error_'+arglist[1]+"_"+arglist[2]+"_"+arglist[3]+"_"+arglist[4]+".txt", 'w') as f:
#     for i in range(min_delta, max_delta):
#          f.write(str(er[i])+'\n')




# ## the data sample too big... try smaller sample size.
# cell_data = data_ed[:n_train]
# cell_target = data_st[:n_train,0]

# ## cross validation
# X_train, X_test, y_train, y_test = \
# cross_validation.train_test_split(cell_data, cell_target, \
#                                   test_size=0.4, random_state=0)

# clf = svm.SVC(kernel='linear', C=1)
# cv_scores = cross_validation.cross_val_score(clf, cell_data, cell_target, cv=5)

# print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))


# import visualize_cells
# reload (visualize_cells)

# ds = data_st.reshape([20,20])
# visualize_cells.visualize_cells(ds)
