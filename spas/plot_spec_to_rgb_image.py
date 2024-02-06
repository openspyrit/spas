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

import math

import numpy as np
from scipy import interpolate

from spas import convert_spec_to_rgb
from spas.convert_spec_to_rgb import ColourSystem
from scipy.ndimage import median_filter as medfilt
from matplotlib import pyplot as plt


def plot_spec_to_rgb_image(GT, wavelengths):
    # cs_srgb = convert_spec_to_rgb.cs_srgb
    cs_hdtv = convert_spec_to_rgb.cs_hdtv
    
    #################### prepare interpolation ################################
    lambda_begin_cie = 380
    lambda_end_cie = 780
    lambda_cie = np.linspace(lambda_begin_cie, lambda_end_cie, int((lambda_end_cie-lambda_begin_cie)/5 +1))
    lambda_begin = math.ceil(wavelengths[0]/5)*5
    lambda_end = math.floor(wavelengths[len(wavelengths)-1]/5)*5
    new_wavelengths = np.arange(lambda_begin, lambda_end+1, 5)
    
    zeros_before_num = int((lambda_begin-lambda_begin_cie)/5)
    zeros_after_num = int((lambda_end_cie-lambda_end)/5)
    zeros_before_vec = np.zeros(zeros_before_num)
    zeros_after_vec = np.zeros(zeros_after_num)
    maxi = np.amax(GT)
    
    image_arr = np.zeros([np.size(GT, axis=0), np.size(GT, axis=1), 3], dtype=np.uint8)
    coeff1 = 0
    for j in range(np.size(GT,axis=1)):
        for i in range(np.size(GT,axis=0)):
            # extract spectrum by spectrum
            pix_spect = GT[i, j, :]
            max_raw_spec = np.amax(pix_spect)
            
            # apply a median filter on each spectrum
            med_spect = medfilt(pix_spect, 99)#pix_spect #
            
            # rescale the spectrum by its original maximum value
            max_med_spec = np.amax(med_spect)
            med_spect = med_spect * max_raw_spec / max_med_spec
            # delete all negative values
            med_spect[np.where(med_spect<0)] = 0
            # interpolate to combine with the spectra from the CIE (cmf) => interpolation degrade the spectral resolution, interpolation of the spectra of the CIE would be better (high resolution)
            f = interpolate.interp1d(wavelengths, med_spect)
            GT_interpol = f(new_wavelengths)
            # complete the spectrum with zeros to multiplicate it with the CIE spectra
            GT_interpol = np.insert(GT_interpol, 0, zeros_before_vec)     
            GT_interpol = np.append(GT_interpol, zeros_after_vec)
            max_GT_interpol = np.amax(GT_interpol)
            if max_GT_interpol != 0:
                GT_interpol = GT_interpol * max_raw_spec / max_GT_interpol
            # find coeff value for rescale rgb vector into the 0-255 range
            coeff = np.amax(GT_interpol)/maxi

            # rgb = ColourSystem.spec_to_rgb(cs_srgb, GT_interpol)
            rgb = ColourSystem.spec_to_rgb(cs_hdtv, GT_interpol)
                
            px = rgb*255*coeff
            image_arr[i, j] = px

            if i == -1:
                print('[i='+str(i)+',j='+str(j)+'] ==> rgb = '+str(image_arr[i, j])+' coeff = '+str(coeff))
    
    
    return image_arr

