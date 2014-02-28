eimport numpy as np
from matplotlib import pyplot as plt
import pdb
import matplotlib
matplotlib.use('QT4Agg')

def visualize_cells (cell):
    "Visualize cells from 20x20 grid cell data."
    k = 0
    cell_coords = np.zeros([400,2])
    cell_exist = np.zeros([400,1])
    for i in range(20):
        for j in range(20):
            cell_coords[k,:] = [i,j] 
            cell_exist[k] = float(cell[i,j])
            k=k+1
                            
    # pdb.set_trace()
    matplotlib.colors.Colormap(cell_exist)
    plt.scatter(cell_coords[:,0],cell_coords[:,1],100,c=cell_exist,marker='s')
    plt.axis((-1,21,-1,21))
    plt.show()
    # plt.get_current_fig_manager().window.raise_()
            





