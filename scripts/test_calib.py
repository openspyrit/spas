# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:36:03 2025

@author: admin
"""
import numpy as np
from spas.acquisition_SPC1D import define_wavelengths_matrix
from matplotlib import pyplot as plt

Lc = [(485, 2)]

acquisition_params.wavelengths = define_wavelengths_matrix(cam_spec_params, Lc)

data_m = np.mean(data[200:240,:], axis = 0)

plt.figure()
plt.plot(acquisition_params.wavelengths[0, :], data_m)