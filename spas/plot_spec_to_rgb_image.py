# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:18:03 2021

@author: mahieu

Converting a spectrum to a colour : https://scipython.com/blog/converting-a-spectrum-to-a-colour/
The formulation used here is based on the CIE (Commission internationnale de l'éclairage 1931) 
colour matching functions, x¯(λ), y¯(λ) and z¯(λ). 
These model the chromatic response of a "standard observer" by mapping a power spectrum of wavelengths, 
P(λ), to a set of tristimulus values, X, Y and Z, analogous to the actual response of the three types 
of cone cell in the human eye.
X=∫P(λ)x¯(λ)dλ
Y=∫P(λ)y¯(λ)dλ
Z=∫P(λ)z¯(λ)dλ

    Args : 
        GT the hyperspectral cube dim : (MxNxLambda)
        wavelengths : the Lambda vector
        
    Retuns :
        image_arr : the image converted into RGB dim : (MxN)x[R G B]

"""

from spas import convert_spec_to_rgb
from spas.convert_spec_to_rgb import ColourSystem
import numpy as np
from matplotlib import pyplot as plt
# import spyrit.misc.walsh_hadamard as wh
# from spas import read_metadata, reconstruction_hadamard
import math
from scipy import interpolate
from scipy.signal import medfilt


def plot_spec_to_rgb_image(GT, wavelengths):
    cs_srgb = convert_spec_to_rgb.cs_srgb
    
    # data_folder_name = '2021-06-01-acquisition-spectral-filter'#'2021-06-23-colored-starsector'#'2021-07-26-spectral-analysis'
    # data_name = 'reconstruction-diffuser-cat2'#'starsector-linearfilter2'#'colored-siemens'
    # ########################### path ###################################
    # data_path = '../data/' + data_folder_name + '/' + data_name# + '/' + data_name
    
    # ########################## read raw data ###########################
    # file = np.load(data_path+'_spectraldata.npz')
    # M = file['spectral_data']#['arr_0']#
    # Q = wh.walsh2_matrix(64)
    
    # metadata_path = data_path + '_metadata.json'
    # metadata, acquisition_parameters, spectrometer_parameters, DMD_parameters = read_metadata(metadata_path)
    
    # GT = reconstruction_hadamard(acquisition_parameters.patterns, 'walsh', Q, M)
    
    # metadata, acquisition_metadata, spectrometer_params, DMD_params = read_metadata(data_path+'_metadata.json')
    # wavelengths = acquisition_metadata.wavelengths
    #################### prepare interpolation ################################
    lambda_begin_cie = 380
    lambda_end_cie = 780
    lambda_begin = math.ceil(wavelengths[0]/5)*5
    lambda_end = math.floor(wavelengths[len(wavelengths)-1]/5)*5
    new_wavelengths = np.arange(lambda_begin, lambda_end+1, 5)
    zeros_before_num = int((lambda_begin-lambda_begin_cie)/5)
    zeros_after_num = int((lambda_end_cie-lambda_end)/5)
    zeros_before_vec = np.zeros(zeros_before_num)
    zeros_after_vec = np.zeros(zeros_after_num)
    maxi = np.amax(GT)
    #print('maxi = '+str(maxi))
    
    image_arr = np.zeros([np.size(GT, axis=0), np.size(GT, axis=1), 3], dtype=np.uint8)
    for j in range(np.size(GT,axis=1)):
        for i in range(np.size(GT,axis=0)):
            pix_spect = GT[i, j, :]
            med_spect = pix_spect#medfilt(pix_spect, 99)
            med_spect[np.where(med_spect<0)] = 0
            gamma = np.max(med_spect)/maxi
            f = interpolate.interp1d(wavelengths, med_spect)
            GT_interpol = f(new_wavelengths)
            GT_interpol2 = np.insert(GT_interpol, 0, zeros_before_vec)            
            GT_interpol3 = np.append(GT_interpol2, zeros_after_vec)

            rgb = ColourSystem.spec_to_rgb(cs_srgb, GT_interpol3)
            
            image_arr[i, j] = rgb*255*gamma
            #print('[i='+str(i)+',j='+str(j)+'] ==> '+image_arr[i, j])  
            
            # new_wavelengths2 = np.arange(lambda_begin_cie, lambda_end+1, 5)
            new_wavelengths3 = np.arange(lambda_begin_cie, lambda_end_cie+1, 5)

            if i == -1:
                print('[i='+str(i)+',j='+str(j)+'] ==> rgb = '+str(image_arr[i, j])+' gamma = '+str(gamma))
                # plt.plot(wavelengths, pix_spect, '-', new_wavelengths3, GT_interpol3, 'o')
                # plt.grid()
                # plt.show()
                # plt.title('[i='+str(i)+',j='+str(j)+'] ==> rgb = '+str(rgb*gamma))
    
    
    return image_arr
    
    # plt.imshow(image_arr, extent=[0, 10.5, 0, 10.5])
    # plt.xlabel('X (mm)')
    # plt.ylabel('Y (mm)')
    # plt.show()





