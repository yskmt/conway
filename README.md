## Machine Learning Exercise for [Reverse the arrow of time in the Game of Life](http://www.kaggle.com/c/conway-s-reverse-game-of-life)

I use primariry scikit-learn Python package to carry out the machine learning process.

* train_data.py - main machine learning process file. Carry out the random forest classification based on the train file in data directory and output the classification results to outs directory.
* gen_paramlist.py - generate a job list based on the output files in outs directory.
* consolidate_prediction.py - consolidate all the output files in outs directory into one Submission.csv file.
* visualize_cells.py - visualize all the cells in 20x20 grid. Need to create a time-dependent visualization.


### Usage of train_data.py

run train_data.py 0 1 0 1
