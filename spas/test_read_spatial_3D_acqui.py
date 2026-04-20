# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:56:41 2026

@author: equipe-onli
"""
from matplotlib import pyplot as plt
import numpy as np

path_name = all_path.raw_data_path + '/' + file_name + '.npz'
raw_data_file = np.load(path_name)
raw_data = raw_data_file['arr_0']
raw_data_file.close()


for i in range(20):
    plt.figure()
    raw = np.rot90(raw_data[:,:,i], k=1, axes=(0,1))
    plt.imshow(raw)
    plt.title('i = ' + str(i))