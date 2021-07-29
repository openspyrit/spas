# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from spyrit.learning.model_Had_DCAN import Hadamard_Transform_Matrix

from singlepixel import read_metadata, reconstruction_hadamard

# Matlab patterns
#file = loadmat('../data/matlab.mat')
#Q = file['Q']

# fht patterns
Q = Hadamard_Transform_Matrix(64)

data_path = '../data/22-04-2021-test-acq/fht_patterns'

file = np.load(data_path+'_spectraldata.npz')
M = file['spectral_data']

metadata, acquisition_metadata, spectrometer_params, DMD_params = read_metadata(data_path+'_metadata.json')
wavelengths = acquisition_metadata.wavelengths

N = 64

frames = reconstruction_hadamard(acquisition_metadata.patterns, 'fht', Q, M)


plt.figure(1)
plt.imshow(frames[:,:,0])
plt.title(f'lambda = {wavelengths[0]:.2f} nm')

plt.figure(2)
plt.imshow(frames[:,:,410])
plt.title(f'lambda = {wavelengths[410]:.2f} nm')

plt.figure(3)
plt.imshow(frames[:,:,820])
plt.title(f'lambda = {wavelengths[820]:.2f} nm')

plt.figure(4)
plt.imshow(frames[:,:,1230])
plt.title(f'lambda = {wavelengths[1230]:.2f} nm')

plt.figure(5)
plt.imshow(frames[:,:,2047])
plt.title(f'lambda = {wavelengths[2047]:.2f} nm')

plt.figure(6)
plt.imshow(np.sum(frames,axis=2))
plt.title('Sum of all wavelengths')