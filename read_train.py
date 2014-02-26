import numpy as np
import csv
from sklearn import svm
from sklearn import cross_validation
import time

# read in  data, parse into training and target sets
dataset = np.genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]

n,m = dataset.shape

# clean up the datasets
id = np.zeros(n)
delta = np.zeros(n)
data_st = np.zeros([n,400])
data_ed = np.zeros([n,400])
for i in range(n):

    id[i] = int(dataset[i,0])
    delta[i] = int(dataset[i,1])

    data_st[i] = np.array(dataset[i,2:402],dtype=int)
    data_ed[i] = np.array(dataset[i,402:802],dtype=int)

## data_ed: features
## data_st[:][0]: class labels of the cell 0

n_train = n/10

start = time.time()

clf = svm.SVC()
clf.fit(data_ed[:n_train], data_st[:n_train,0])


end = time.time()
print "ellapsed time = ", end - start

## the data sample too big... try smaller sample size.
cell_data = data_ed[:n_train]
cell_target = data_st[:n_train,0]

## cross validation
X_train, X_test, y_train, y_test = \
cross_validation.train_test_split(cell_data, cell_target, \
                                  test_size=0.4, random_state=0)

clf = svm.SVC(kernel='linear', C=1)
cv_scores = cross_validation.cross_val_score(clf, cell_data, cell_target, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))


# import visualize_cells
# reload (visualize_cells)

# ds = data_st.reshape([20,20])
# visualize_cells.visualize_cells(ds)

