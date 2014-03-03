## Conway's Reverse Game of Life in [Keggle Competition](http://www.kaggle.com/c/conway-s-reverse-game-of-life).

#####Conway's Game of Life explanation is [here](http://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).

#####Reverse Game of Life objective: using the final states of the cells in 20*20 grid (= 400 cells) and how many generation has been passed, determine the initial state of each cell.
___

####Strategy

I use primariry scikit-learn Python package to carry out the machine learning process.

1. Consider that we have 20*20*5=2000 models. There exists 400 different cells and 5 different "depth". Using the final states of the cell (400 features), classify whether the initial state of the cell is alive or dead.
2. Training data can be obtianed from the competition [website](http://www.kaggle.com/c/conway-s-reverse-game-of-life/data).
3. Use random forest classifiyer algorithm from scikit-learn.

####Files

* train_data.py - main machine learning process file. Carry out the random forest classification based on the train file in data directory and output the classification results to outs directory.
* gen_paramlist.py - generate a job list based on the output files in outs directory.
* consolidate_prediction.py - consolidate all the output files in outs directory into one Submission.csv file.
* visualize_cells.py - visualize all the cells in 20x20 grid. Need to create a time-dependent visualization.


#### Usage of train_data.py

run train_data.py [depth] [depth+1] [cell#] [cell#+1]

##### Example:
run train_data.py 0 1 0 1

