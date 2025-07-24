# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:13:57 2025

@author: admin
"""

import pickle
import numpy as np
from spas.acquisition_SPC1D import read_metadata
from matplotlib import pyplot as plt
import os
os.chdir('C:\\openspyrit\\spas\\scripts')
from scipy.signal import find_peaks

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
#%% read data
Lc_acq = 912 # 811 #  795 #696 #  546 #436 # 405 # 365 # 577 #763 #842 #     
Gr = 1
if Lc_acq == 365:
    Lc_real = 365.015
elif Lc_acq == 405:
    Lc_real = 404.656
elif Lc_acq == 436:
    Lc_real = 435.833
elif Lc_acq == 546:
    Lc_real = 546.074
elif Lc_acq == 577:
    Lc_real = 576.96   
elif Lc_acq == 696:
    Lc_real = 696.543
elif Lc_acq == 763:
    Lc_real = 763.511
elif Lc_acq == 795:
    Lc_real = 794.818
elif Lc_acq == 811:
    Lc_real = 811.531    
elif Lc_acq == 842:
    Lc_real = 842.465
elif Lc_acq == 912:
    Lc_real = 912.297
    
if Gr == 1:    
    # data_folder_name = '2025-07-10_wavelength-calibration'
    data_folder_name = '2025-07-24_wavelength-calibration-Gr1'
elif Gr == 2:
    data_folder_name = '2025-07-21_wavelength-calibration-Gr2'
    
folder_path = os.listdir('../../data/' + data_folder_name)
substring = 'Ray-' + str(Lc_acq)

inc = 0    
for phrase in folder_path: 
    if substring in phrase:
        indx = inc
    inc = inc + 1

indx1 = folder_path[indx].index('_ti_')
indx2 = folder_path[indx].index('ms_')
ti = folder_path[indx][indx1 + 4: indx2]
    
data_name = 'obj_Ray-'  + str(Lc_acq) + '_source_HG-1_Oceanoptics_Walsh_im_2x2_ti_' + str(ti) + 'ms_zoom_x1'

output_path = '../../data/' + data_folder_name + '/' + data_name
saved_DMD_params, saved_spectrograph_params, saved_cam_spat_params, saved_cam_spec_params, saved_acquisition_params = read_metadata(output_path + '/metadata.json')

iNR = 0
iNA = 0
iNp = 0
Nx = saved_cam_spec_params.width
Ny = saved_cam_spec_params.height
Lc = saved_acquisition_params.Lc
NLc = len(Lc)

# exeption
if Gr == 2 and (Lc_acq == 763 or Lc_acq == 811):
    del Lc[9]
    NLc = len(Lc)

spectral_data_all2 = np.empty((Ny, Nx, NLc), dtype = float)
for iLc in range(NLc):
    data_path = output_path + '/raw_data/spectral_NR_' + str(iNR) + '_Gr_' + str(Lc[iLc][1]) + '_Lc_' + str(Lc[iLc][0]) + 'nm_NA_' + str(iNA) + '_NS_' + str(iNp) + '.pkl'
    
    with open(data_path, "rb") as fp:
        pickle_image = pickle.load(fp)

    spectral_data_all2[:, :, iLc] = pickle_image

Lambda = np.empty(NLc - 1)
for i in range(NLc - 1):
    Lambda[i] = saved_acquisition_params.Lc[i + 1][0]
#%% exploit data for Gr = 1    
display_fig_nbr = 1
pic = np.empty(NLc)
width = np.empty(NLc)

for iLc in range(NLc):
    if iLc == display_fig_nbr:
        plt.figure()
        plt.imshow(spectral_data_all2[:, :, iLc])
        plt.title('Lambda = ' + str(Lc[iLc][0]) + ' - Gr + ' + str(Lc[iLc][1]))

    # profile = np.mean(spectral_data_all[290:370, :, iLc], axis = 0)
    profile_init = spectral_data_all2[220, :, iLc]
    profile = profile_init - 126
    # if Gr == 1 and iLc == 1:
    #     profile = profile * 3
    profile[profile < 0] = 0
    profile[profile > 220] = 220
    
    if iLc == display_fig_nbr:
        # plt.figure()
        # plt.plot(profile_init)
        # plt.title('Lambda = ' + str(Lc[iLc][0]) + ' - Gr + ' + str(Lc[iLc][1]))
        
        plt.figure()
        plt.plot(profile)
        plt.title('Lambda = ' + str(Lc[iLc][0]) + ' - Gr + ' + str(Lc[iLc][1]))
    
    width_peak = 10
    height_peak = 150
    # if Gr == 1 and iLc == 5 and Lc_acq == 436:
    #     height_peak = 100
    # elif Gr == 1 and iLc == 5 and Lc_acq == 546:
    #     print('ici')
    #     height_peak = 50
    #     width_peak = 5
    # else:
    #     height_peak = 150
        
        
    if Gr == 2:
        dec = 65        
    elif Gr == 1:
        dec = 65#134
        
    peaks, properties = find_peaks(profile, height = height_peak, width = width_peak)
    
    if 1==1:#Gr == 2:
        if len(peaks > 1):
            indx = find_nearest(Lambda, Lc_acq)
            value = (indx + 1 - iLc) * dec + 640
            indx = find_nearest(peaks, value)
        else:
            indx = 0
            
        pic[iLc] = peaks[indx]
        # print(peaks)
        # print(pic)
        width[iLc] = properties['widths'][indx]
        
    if 1==2:#Gr == 1:
        # if Gr == 2 and Lc_acq == 405 and (iLc > 0 and iLc < 8):
        #     pic[iLc] = peaks[1]
        #     width[iLc] = properties['widths'][1]
        if Lc_acq == 436 and iLc == 1:
            pic[iLc] = peaks[1]
            width[iLc] = properties['widths'][1]
        elif Lc_acq == 546 and iLc == 5:
            peaks, properties = find_peaks(profile, height = 120, width = 10)
            pic[iLc] = peaks[0]
            width[iLc] = properties['widths'][0]
        elif Lc_acq == 577 and (iLc == 1 or iLc == 2):
            pic[iLc] = peaks[1]
            width[iLc] = properties['widths'][1]
        elif Lc_acq == 696 and (iLc == 1 or iLc == 5):
            peaks, properties = find_peaks(profile, height = 100, width = 10)
            pic[iLc] = peaks[0]
            width[iLc] = properties['widths'][0]
        elif Lc_acq == 795 and iLc < 3 and iLc != 0:
            if iLc == 1:
                peaks, properties = find_peaks(profile, height = 150, width = 10)
                pic[iLc] = peaks[2]
                width[iLc] = properties['widths'][2]
            elif iLc == 2:
                # peaks, properties = find_peaks(profile, height = 150, width = 10)
                pic[iLc] = peaks[1]
                width[iLc] = properties['widths'][1]
        else:       
            pic[iLc] = peaks[0]
            width[iLc] = properties['widths'][0]
        
    print('iLc = ' + str(iLc))
    print('peaks = ' + str(peaks))
    print('pic = ' + str(pic[iLc]))
    # if iLc == 1:
    #     break

    # if iLc == 8:
    #     L_v = [404.656, 435.833, 546.074, 576.96]
        
    #     z = np.polyfit(L_v, peaks, 2)
    #     p = np.poly1d(z)
        
    #     x = np.linspace(L_v[0], L_v[-1], 1000)
        
    #     droite = p[2] * x**2 + p[1] * x + p[0]
        
    #     plt.figure()
    #     plt.plot(L_v, peaks, '*r', x, droite)
    #     plt.xlabel('Lambda (nm)')
    #     plt.ylabel('pixel')
    #     plt.grid()
        
zero_px = pic[0]
print('zero set to : ' + str(zero_px) + ' px')
pic=pic[1:] # delete the pic for lambda = 0 nm
width = width[1:] # delete the width for lambda = 0 nm

if Gr == 1 and Lc_acq == 795:
    pic = pic[:-2]
    width = width[:-2]
    Lambda = Lambda[:-2]
    NLc = NLc - 2
    
print('width = ' + str(width))
width_mean = np.mean(width)
print('     mean width = ' + str(width_mean) + ' px')

dispersion_px = np.empty(NLc - 1)
dispersion_lambda = np.empty(NLc - 1)
for i in range(len(pic - 1)):
    dispersion_px[i] = pic[i] - zero_px
    dispersion_lambda[i] = Lc_real - Lambda[i]

dispersion = dispersion_lambda / dispersion_px
dispersion = np.delete(dispersion, 2)
print('dispersion by pixel = ' + str(dispersion) + ' nm/px')
dispersion_mean = np.mean(dispersion)
print('     mean dispersion by pixel = ' + str(dispersion_mean) + ' nm/px')

resolution = width * dispersion_mean
print('resolution = ' + str(resolution) + ' nm')
resolution_mean = width_mean * dispersion_mean
print('     mean resolution = ' + str(resolution_mean) + ' nm')

# coeff_dir = (dispersion_px[-1] - dispersion_px[0]) / (dispersion_lambda[-1] - dispersion_lambda[0])

z = np.polyfit(dispersion_lambda, dispersion_px, 1)
p = np.poly1d(z)

coeff_dir = p[1]

print('coeff dir = ' + str(coeff_dir))
print('ord à l''origine = ' + str(p[0]))

plt.figure()
plt.plot(dispersion_lambda, dispersion_px)
plt.grid()
plt.xlabel('Lambda (nm)')
plt.ylabel('pixel')
plt.show()

wavelength_matrix = np.empty((NLc - 1, 1280))

for iLc in range(NLc - 1):
    limit_inf = (0 - pic[iLc]) / coeff_dir
    limit_sup = (1279 - pic[iLc]) / coeff_dir
    
    x = np.linspace(limit_inf, limit_sup, 1280)
    
    droite = coeff_dir * x + p[0]

    if iLc == round(NLc/2):
        plt.figure()
        plt.plot(dispersion_lambda, dispersion_px, 'x')
        plt.plot(x, droite)
        plt.xlabel('Lambda (nm)')
        plt.ylabel('pixel')
        plt.grid()
        plt.show()
    
    dx = droite[2] - droite[1]
    dL = x[2] - x[1]
    
    wavelength_vector = x + Lambda[round(len(Lambda)/2)]
    wavelength_matrix[iLc, :] = wavelength_vector
#%% wavelength vector
wavelength_matrix = np.empty((NLc - 1, 1280))

for iLc in range(NLc - 1):
    pici = 640 + p[0] + (405 - Lambda[iLc]) * p[1]
    limit_inf = (0 - pici) / coeff_dir
    limit_sup = (1279 - pici) / coeff_dir
    
    x = np.linspace(limit_inf, limit_sup, 1280)
    
    wavelength_vector = x + Lambda[round(len(Lambda)/2)]
    wavelength_matrix[iLc, :] = wavelength_vector
#%% dips
disp = np.empty(len(dispersion_px) - 1)
for i in range(len(dispersion_px) - 1):
    disp[i] = dispersion_px[i] - dispersion_px[i + 1]
print(disp)
#%% plot result
display_fig_nbr = 3
for iLc in range(NLc - 1):
    profile_init = spectral_data_all2[220, :, iLc + 1]
    
    if 1==1:#iLc == display_fig_nbr:
        plt.figure()
        plt.plot(wavelength_matrix[iLc, :], profile_init)
        plt.title('Lambda = ' + str(Lc[iLc + 1][0]) + ' - Gr + ' + str(Lc[iLc + 1][1]))
        plt.xlabel('nm')        
#%% final fit
from scipy import interpolate

# Gr = 1
# Lc_array = np.array([365, 405, 436, 546, 577, 696, 795])
# coeff_array = np.array([14, 14.075, 14.3, 14.82, 14.3, 15.75, 16.63])

# Lc_array = np.array([365, 405, 436, 546, 696, 795])
# coeff_array = np.array([14, 14.075, 14.3, 14.82, 15.75, 16.63])

# Lc_array = np.array([365, 405, 436, 546, 696, 795])
# coeff_array = np.array([13.8, 14.21, 14.44, 15.2, 15.75, 16.63])
# ord_origine = np.array([-1, -0.31, -6.99, -4.92,
    
# Lc_array = np.array([   365,    405,    436,   546,    696, 795])
# coeff_array = np.array([13.82, 14.03, 14.28, 14.92,  15.81, 16.63])
# ord_origine = np.array([-0.21,  0.63, -5.41,  -2.5, -10.88,

Lc_array = np.array([     365,   405,   436,   546,    696,   912])
coeff_array = np.array([13.26, 14.18, 14.12, 14.94,  15.89, 18.36])
ord_origine = np.array([ 3.94, -0.69, -6.5,  -3.53,  -9.63, -5.88])
                        
z = np.polyfit(Lc_array, coeff_array, 2)
p = np.poly1d(z)

# xnew = np.arange(Lc_array[0], Lc_array[-1])
xnew = np.arange(200, 1000)
plt.figure()
plt.plot(Lc_array, coeff_array, '.', xnew, p(xnew), '-')
plt.grid()
plt.title('GR = 1')

# Gr = 2
# Lc_array = np.array([365, 405, 435, 546, 577, 763, 842])
# # coeff_array = np.array([3.219, 3.231, 3.231, 3.225, 3.25, 3.269, 3.269])
# coeff_array = np.array([3.2275, 3.2566, 3.2783, 3.2566, 3.2733, 3.3066, 3.254])
# ord_origine = np.array([2.9516, 3.787, -3.064, -1.13, 2.5754, 6.977, 3.376])

Lc_array = np.array([365, 405, 435, 546, 577, 696, 763, 811, 912])
coeff_array = np.array([3.308, 3.342, 3.311, 3.319, 3.324, 3.341, 3.389, 3.304, 3.365])
ord_origine = np.array([-5.5, -6.63, -9.56, -7.58, -3.09, -6.92, -4.87, -8.42, -3.89])

z = np.polyfit(Lc_array, coeff_array, 2)
p = np.poly1d(z)

# xnew = np.arange(Lc_array[0], Lc_array[-1])
xnew = np.arange(200, 1000)
plt.figure()
plt.plot(Lc_array, coeff_array, '.', xnew, p(xnew), '-')
plt.grid()
plt.title('GR = 2')

Lc_array = np.array([365, 436, 546, 577, 696, 912])
coeff_array = np.array([3.308, 3.311, 3.319, 3.324, 3.341, 3.365])
ord_origine = np.array([-5.5, -9.56, -7.58, -3.09, -6.92, -3.89])

z = np.polyfit(Lc_array, coeff_array, 2)
p = np.poly1d(z)

# xnew = np.arange(Lc_array[0], Lc_array[-1])
xnew = np.arange(200, 1000)
plt.figure()
plt.plot(Lc_array, coeff_array, '.', xnew, p(xnew), '-')
plt.grid()
plt.title('GR = 2')


z = np.polyfit(Lc_array, ord_origine, 10)
p = np.poly1d(z)

# xnew = np.arange(Lc_array[0], Lc_array[-1])
xnew = np.arange(200, 1000)
plt.figure()
plt.plot(Lc_array, ord_origine, '.', xnew, p(xnew), '-')
plt.grid()
plt.title('GR = 2')

# # 365
# x0 = np.linspace(350,405)
# y0 = coeff_array[0] * x + ord_origine[0]
# #405
# x1 = np.linspace(385,435)
# y1 = coeff_array[1] * x + ord_origine[1]
# #435
# x2 = np.linspace(420,546)
# y2 = coeff_array[2] * x + ord_origine[2]
# #546
# x3 = np.linspace(490,577)
# y3 = coeff_array[3] * x + ord_origine[3]
# #577
# x4 = np.linspace(561,696)
# y4 = coeff_array[4] * x + ord_origine[4]
# #763
# x5 = np.linspace(561,696)
# y5 = coeff_array[5] * x + ord_origine[5]
# #842
# x6 = np.linspace(802,900)
# y6 = coeff_array[6] * x + ord_origine[6]

# plt.figure()
# plt.plot(x0, y0,'r', x1, y1,'g', x2, y2,'blue', x3, y3,'black', x4, y4,'c', x5, y5,'y', x6, y6,'m',)

# f = interpolate.interp1d(Lc_array, coeff_array)

# xnew = np.arange(Lc_array[0], Lc_array[-1])

# ynew = f(xnew)

# plt.figure()
# plt.plot(Lc_array, coeff_array, 'o', xnew, ynew, '-')
# plt.show()
#%% 
from scipy import interpolate

display_figure = 1
iLc = 8
Lcc = Lc[iLc+1][0]
Grating = Lc[iLc+1][1]

if Grating == 2:
    Lc_array = np.array([365, 436, 546, 577, 696, 912])
    coeff_array = np.array([3.308, 3.311, 3.319, 3.324, 3.341, 3.365])
    # ord_origine = np.array([-5.5, -9.56, -7.58, -3.09, -6.92, -3.89])
    ord_origine = np.array([-5.5, -9.56, -7.58, -3.09, -6.92, -3.89])
elif Grating == 1:
    Lc_array = np.array([365, 405, 436, 546, 696, 795])
    coeff_array = np.array([14, 14.075, 14.3, 14.82, 15.75, 16.63])

z = np.polyfit(Lc_array, coeff_array, 2)
p = np.poly1d(z)

# xnew = np.arange(Lc_array[0], Lc_array[-1])
xnew = np.arange(200, 1000)
if display_figure:
    plt.figure()
    plt.plot(Lc_array, coeff_array, '.', xnew, p(xnew), '-')
    plt.grid()
    plt.title('GR = 2')

indx = find_nearest(xnew, Lcc)

px_width = 1280#cam_spec_params.width
px_offset = 0#cam_spec_params.offsetX

# spam_lambda = px_width / np.mean(p(xnew))
spam_lambda = px_width / p(xnew[indx])

# L_offset = px_offset / np.mean(p(xnew))
L_offset = px_offset / p(xnew[indx])
L1 = Lcc - spam_lambda / 2 + L_offset
L2 = Lcc + spam_lambda / 2 + L_offset

xnew2 = np.linspace(L1, L2, px_width)
if display_figure:
    plt.figure()
    plt.plot(Lc_array, coeff_array, '.', xnew2, p(xnew2), '-')
    plt.grid()
    plt.title('GR = 2')

f = interpolate.interp1d(Lc_array, ord_origine, fill_value='extrapolate')

ynew = f(xnew2)
plt.figure()
plt.plot(Lc_array, ord_origine, '.', xnew2, ynew, 'r')

w = np.empty(px_width)
for i in range(len(w)):
    # w[i] = (i - 640 - ynew[i]) / p(xnew2[i]) + Lcc + L_offset
    w[i] = (i - 640) / p(xnew2[i]) + Lcc + L_offset
    
    if i == 0 or  i == len(w) - 1 :
        print('i = ' + str(i) + ' => pente = ' + str(p(xnew2[i])) + ' => ord = ' + str(ynew[i]))
    # w[i] = (i - 640) / p(xnew2[i]) + Lcc + L_offset
        
        
#%% plot result
display_fig_nbr = 3
# for iLc in range(NLc - 1):
profile_init = spectral_data_all2[220, :, iLc + 1]

if 1==1:#iLc == display_fig_nbr:
    plt.figure()
    # plt.plot(wavelength_matrix[iLc, :], profile_init)
    plt.plot(w, profile_init)
    plt.title('Lambda = ' + str(Lc[iLc + 1][0]) + ' - Gr + ' + str(Lc[iLc + 1][1]))
    plt.xlabel('nm')            
    
        
        
        
        
        
        
        
        
        
        