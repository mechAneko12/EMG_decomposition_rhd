import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_emg(ndarray):
    for i in range(ndarray.shape[1]):
        ndarray[:, i] = ndarray[:, i] + i*0.2
    print(np.max(ndarray))
    pd.DataFrame(ndarray).plot(legend=False)
    plt.show()
    
if __name__ == '__main__':
    ndarray = pd.read_csv('dataset/20220712/exp0712-2.csv').values[:75000, :32] * 0.0001
    print(np.max(ndarray))
    plot_emg(ndarray)
