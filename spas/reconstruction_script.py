# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from singlepixel import *

file = loadmat('./data/matlab.mat')
Q = file['Q']

data_path = './data/01-06-2021-acquisition-spectral-filter/reconstruction-diffuser-cat2'

data = np.load(data_path+'_spectraldata.npz')
M = data['spectral_data']
#wavelenghts = np.loadtxt(data_path+'_lambda.txt')
#np.savetxt(data_path+'spectraldata.csv',M,delimiter=",")

metadata, acquisition_metadata, spectrometer_params, DMD_params = read_metadata(data_path+'_metadata.json')
wavelenghts = acquisition_metadata.wavelengths

N = 64

ind_opt = acquisition_metadata.patterns[1::2]
ind_opt = np.array(ind_opt)/2
ind_opt = ind_opt - 1
ind_opt = ind_opt.astype('int')
M_breve = M[0::2,:] - M[1::2,:]
M_Had = np.zeros((N*N, M.shape[1]))
M_Had[ind_opt,:] = M_breve


f = np.matmul(Q,M_Had) # Q.T = Q (Q transposed is Q)
#frame = np.reshape(f[:,1024],(-1,N))
frames = np.reshape(f,(N,N,M.shape[1]))

frames2 = frames.T
from singlepixel.visualization import spectral_binnning, plot_color

F_bin, w_bin, bin_width = spectral_binnning(frames2, wavelengths, 530, 730, 8)

plot_color(F_bin, wavelengths)

"""
plt.figure(1)
plt.imshow(frames[:,:,0])
plt.title(f'lambda = {wavelenghts[0]:.2f} nm')

plt.figure(2)
plt.imshow(frames[:,:,410])
plt.title(f'lambda = {wavelenghts[410]:.2f} nm')

plt.figure(3)
plt.imshow(frames[:,:,820])
plt.title(f'lambda = {wavelenghts[820]:.2f} nm')

plt.figure(4)
plt.imshow(frames[:,:,1230])
plt.title(f'lambda = {wavelenghts[1230]:.2f} nm')

plt.figure(5)
plt.imshow(frames[:,:,2047])
plt.title(f'lambda = {wavelenghts[2047]:.2f} nm')

plt.figure(6)
plt.imshow(np.sum(frames,axis=2))
plt.title('Sum of all wavelenghts')"""