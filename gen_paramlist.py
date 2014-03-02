import os


with open('paramlist_outs', "w") as f:
	 for depth in range(5):		 
	 	 for model_num in range(400):
			 outfile = "outs/out_{0:d}_{1:d}_{2:d}_{3:d}.txt".format(depth,depth+1,model_num,model_num+1)
			 if not os.path.isfile(outfile):
				 print outfile+" does not exist"
				 f.write("python read_train.py {0:d} {1:d} {2:d} {3:d}\n".format(depth, depth+1, model_num, model_num+1))
		
