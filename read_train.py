import numpy as np
import csv
from sklearn import svm
from sklearn import cross_validation
import time
from sklearn.externals import joblib

def svm_vectors ( train, target, dim ):
    "Get the SVN vectors for each cell"
    clfs = []
    start = time.time()
    for i in range(dim):
        print "%dth iteration" %i
        st_svc = time.time()
        clfs.append(svm.SVC())
        clfs[i].fit(train, target[:,i])
        ed_svc = time.time()
        print "done, time: %f" %(ed_svc-st_svc)        
    end = time.time()

    print "ellapsed time = ", end - start

    return clfs

def svm_predict (clfs, data_test, dim ):
    "Predict the values for each cell"
    n_train, _ = data_test.shape
    data_predict = np.zeros([dim, n_train])

    for i in range(dim):
        data_predict[i] = np.array(clfs[i].predict( data_test) )

    return data_predict.T


# read in  data, parse into training and target sets
print "reading the dataset"
dataset = np.genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
n,m = dataset.shape

# clean up the datasets
print "cleaning up the dataset"
id = np.zeros(n)
delta = np.zeros(n)
data_st = np.zeros([n,400])
data_ed = np.zeros([n,400])
for i in range(n):
    id[i] = int(dataset[i,0])
    delta[i] = int(dataset[i,1])

    data_st[i] = np.array(dataset[i,2:402],dtype=int)
    data_ed[i] = np.array(dataset[i,402:802],dtype=int)

# run the svm
print "runnning the svm"
_, dim = data_st.shape
# dim = 10
n_train = n/5
clfs = svm_vectors( data_ed[:n_train], data_st[:n_train,:], dim)

# save the clfs
print "saving the clfs"
joblib.dump(clfs, 'conway.pkl') 

# predict the values
print "predicting the values"
data_predict = svm_predict( clfs, data_ed[n_train:n_train*2-1], dim)


error = sum((abs(data_st[n_train:n_train*2-1,0:dim] - data_predict)).T)
print "error % = %f" %(sum(error)/((n_train-1)*dim))

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
