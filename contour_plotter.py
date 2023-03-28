import _file_reader as fr
import matplotlib.pyplot as plt
import numpy as np


def cont_plotter(filname):
    lines = fr.file_reader(filname)
    for i, line in enumerate(lines):
        points = line[0]
        arr = np.array(points)
        plt.plot(arr[:,0], arr[:,1], 'k')
    plt.show()


cont_plotter("myfile.txt")