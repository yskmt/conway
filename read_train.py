import numpy as np
import csv
from sklearn import svm
from sklearn import cross_validation
import time
from sklearn.externals import joblib
import pdb

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
        print "%dth iteration" %i
        st_svc = time.time()
        data_predict[i] = np.array(clfs[i].predict( data_test) )
        ed_svc = time.time()
        print "done, time: %f" %(ed_svc-st_svc)        

    return data_predict.T

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


n_div = 5


# read in  data, parse into training and target sets
print "reading the dataset"
dataset = np.genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
n,m = dataset.shape

# clean up the datasets
print "cleaning up the dataset"
id = np.zeros(n, dtype=int)
delta = np.zeros(n, dtype=int)
# data_st = np.zeros([n,400], dtype=int)
# data_ed = np.zeros([n,400], dtype=int)
data_st = [[] for i in range(5) ]
data_ed = [[] for i in range(5) ]
for i in range(n):
    id[i] = int(dataset[i,0])
    delta[i] = int(dataset[i,1])
    
    data_st[delta[i]-1].append(dataset[i,2:402])
    data_ed[delta[i]-1].append(dataset[i,402:802])
    # data_st[i] = np.array(dataset[i,2:402],dtype=int)
    # data_ed[i] = np.array(dataset[i,402:802],dtype=int)

for i in range(max(delta)):
    data_st[i] = np.array(data_st[i], dtype=int)
    data_ed[i] = np.array(data_ed[i], dtype=int)

# classification functions for [delta][cell#]
clfs = [[] for i in range(max(delta))]
# solve the problem for different levels
for i in range(max(delta)):

    # run the svm
    print "runnning the svm for delta = %i" %(i+1)
    n_train, dim = data_st[i].shape
    n_train = n_train/n_div
    clfs[i] =  svm_vectors( data_ed[i][:n_train], data_st[i][:n_train], dim)

    # save the clfs
    print "saving the clfs for delta = %i" %(i+1)
    joblib.dump(clfs, 'conway_%i.pkl'%i) 

# predicted data for [delta]
data_predict = [[] for i in range(max(delta))]
# predict the values for different levels
for i in range(max(delta)):
    print "predicting the values for delta = %i" %(i+1)
    n_train, dim = data_st[i].shape
    n_train = n_train/n_div
    data_predict[i] = svm_predict( clfs[i], data_ed[i][n_train:n_train*2+1], dim)

er = np.zeros(max(delta))
for i in range(max(delta)):
    n_train, dim = data_st[i].shape
    n_train = n_train/n_div
    er[i] = sum(sum(abs(data_st[i][n_train:n_train*2+1,0:dim] - data_predict[i]).T)) / \
        ((n_train+1)*dim)
    # er[i] = sum((abs(data_st[i][n_train:n_train*2+1,0:dim] - data_predict[i])).T)
    # print "error %% = %f" %(sum(er[i])/((n_train+1)*dim))

print er




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
