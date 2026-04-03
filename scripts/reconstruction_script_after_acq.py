# -*- coding: utf-8 -*-
__author__ = 'Laurent Mahieu-Williame'

#%% Package

import spyrit.misc.walsh_hadamard as wh
from spas.visualization import plot_reco_without_NN, plot_reco_with_NN 
from spas.metadata_SPC2D import read_metadata, read_metadata_2arms, func_path
from spas.transfer_data_to_girder import transfer_data_2arms, transfer_data
from spas.reconstruction_nn import ReconstructionParameters, setup_reconstruction
from spas.reconstruction_SPC2D import reconstruction_hadamard

import os
import numpy as np
import time    
import pickle
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
os.chdir("d:/hspc/scripts")
#%% INPUT
delete_old_fig = 0
plot_had_reco = 1
nn_reco = 1
plot_nn_reco = 1
tranfer = 0
write_in_csv_file = 0
process_data = 1
write_result_in_csv_file = 0
fit_loop_nbr = 2 # to fit nn reco => put 2 ||| to fit had reco => put 1
sub_file = 'GT'#'spectral_data' # 
patient_num = 62
substract_with_no_light = 0
masking = 1
#%% Begin
plt.close('all')
t_tot_0 = time.time()
############################ CSV file ##########################
csv_file_path = 'data_in_Girder/data.csv'
csv_exist = os.path.isfile(csv_file_path)
fieldnames = ['setup_version', 'data_folder_name', 'data_name', 'transfered_to_girder', 'had_reco', 'nn_reco', 'check_data_exist_in_girder', 'delete_old_fig']

csv_result_file_path = '../result/hopital/fit_result_two_curve.csv'
csv_result_exist = os.path.isfile(csv_result_file_path)
# result_fieldnames = ['data_folder_name', 'data_name', 'offset', 'Magnitude', 'Center (nm)', 'sigma', 'Area', 'image size', 'ti']
result_fieldnames = ['data_folder_name', 'data_name', 'reco', 'offset', 'Amp 1', 'Lambda 1', 'sigma 1', 'Amp 2', 'Lambda 2', 'sigma 2', 'Area', 'image size', 'ti']
########################## to be change ############################
setup_version = 'setup_v1.3.1'
collection_acces = 'private'    #publique
########################## PATH ##################################
root_path = 'D://hspc//data//'
folder_name_begging = 'Patient-' + str(patient_num) + '_exvivo_'
if patient_num <= 63:
    add_BU = ''
else:
    add_BU = '_BU'
stop = 0
if os.path.exists(root_path + folder_name_begging + 'LGG' + add_BU) == True:
    grade = 'LGG'
elif os.path.exists(root_path + folder_name_begging + 'HGG' + add_BU) == True:
    grade = 'HGG'   
elif os.path.exists(root_path + folder_name_begging + 'meningioma' + add_BU) == True:
    grade = 'meningioma'
else:
    print('patient folder does not exist')
    stop = 1
    grade = ''

data_folder_name = folder_name_begging + grade + add_BU
# data_folder_name = 'Patient-64_exvivo_LGG'

temp_data_file_list = os.listdir('../data/' + data_folder_name)
data_file_list = temp_data_file_list#[0]

#data_file_list = ['obj_biopsy-8_deep-anterior-limit_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1'] # patient 60
# data_file_list = ['obj_biopsy-7-intern-limit-GP_source_Laser_405nm_1.2W_A_0.15_f80mm-P2_Walsh_im_16x16_ti_150ms_zoom_x1'] # patient 61
data_file_list = ['obj_biopsy-7_posterior-part_source_Laser_405nm_1.2W_A_0.14_f80mm-P2_Walsh_im_64x64_ti_5ms_zoom_x1'] # patient 62
# data_file_list = ['obj_biopsy-7-anterior-limit_source_Laser_405nm_1.2W_A_0.14_f80mm-P2_Walsh_im_16x16_ti_100ms_zoom_x1'] # patient 64


# data_file_list.remove('obj_biopsy-1_tumor_center_source_385nm_Walsh_im_16x16_ti_200ms_zoom_x1')
# data_file_list.remove('obj_biopsy-1_tumor_center_source_405nm_Walsh_im_32x32_ti_200ms_zoom_x1')


############################ Beginning ##########################
inc = 0
# data_name = data_file_list[0]
for data_name in data_file_list:
    ########################### path ###################################
    print('data folder : '+data_folder_name)
    print('  --- data name : ' + data_name)  
    all_path = func_path(data_folder_name, data_name)
    ######################## source ###################################
    indx_source1 = data_name.find('source')
    indx_source1 = indx_source1 + len('source') + 1 
    indx_source2 = data_name[indx_source1:].find('_f80mm')
    source = data_name[indx_source1:indx_source1 + indx_source2]
    ############### find the No-Light file for substraction #############
    data_name_no_ligth = data_name[:indx_source1] + 'No-light' + data_name[indx_source1 + indx_source2:]
    sub_folder_exist = os.path.exists('../data/' + data_folder_name + '/' + data_name_no_ligth)
    if sub_folder_exist == True:
        if substract_with_no_light == 1:
            substraction = 1
        else:
            substraction = 0        
    else:
        substraction = 0
        print('------ No-light file found !!! ------')
    ############### find the LED file for masking #######################
    ti = 100
    data_name_white_LED = data_name[:indx_source1] + 'white_LED' + data_name[indx_source1 + indx_source2:]
    indx_source3 = data_name_white_LED.find('_ti_')
    indx_source3 = indx_source3 + len('_ti_')
    indx_source4 = data_name_white_LED.find('ms_zoom_')
    
    data_name_white_LED2 = data_name_white_LED[:indx_source3] + str(ti) + data_name_white_LED[indx_source4:]
    full_data_name_white_LED2 = '../data/' + data_folder_name + '/' + data_name_white_LED2 
    
    sub_folder_exist = os.path.exists(full_data_name_white_LED2)
    if sub_folder_exist == True:
        print('white LED file found')
        white_field_file = np.load(full_data_name_white_LED2 + '/' +  data_name_white_LED2 + '_nn_reco.npz')
        white_field = white_field_file['arr_0']
        
        white_field = np.rot90(white_field, 2, axes=(1,0))
        white_field = np.rot90(white_field, 2, axes=(1,2))
        white_field = np.flip(white_field, axis=2)
        
        plt.figure()
        plt.imshow(white_field[:,:,800:1200].sum(axis=2))
        
        plt.figure()
        plt.imshow(white_field.sum(axis=2))
    else:
        print('white LED file not found')
    ###################### delete old figures ###########################
    if delete_old_fig == 1:
        fig_list = os.listdir(all_path.overview_path)
        for fig in fig_list:
            if fig.find('HAD_RECO') >=0 or fig.find('NN_RECO') >=0:
                print(fig)
                os.remove(all_path.overview_path + '/' + fig)
    ########################### read metadata ###########################
    metadata_path = all_path.data_path + '_metadata.json'
    # metadata, acquisition_parameters, spectrometer_parameters, DMD_parameters = read_metadata(metadata_path)
    metadata, acquisition_parameters, spectrometer_parameters, DMD_parameters, camPar  = read_metadata_2arms(metadata_path)

    wavelengths = acquisition_parameters.wavelengths
    meas_size = acquisition_parameters.pattern_dimension_x * acquisition_parameters.pattern_dimension_y * 2
    Np = acquisition_parameters.pattern_dimension_x  
    ti = spectrometer_parameters.integration_time_ms
    ########################## laod spectral data matrix ################
    file = np.load(all_path.data_path + '_spectraldata.npz')
    try:
        spectral_data = file['spectral_data']
        # print('npz item : spectral_data')
        # for k in file.data_name:
        #     print(k)
    except:
        spectral_data = file['arr_0']
        print('npz item : arr_0')
    file.close()
    if substraction == 1 and sub_file == 'spectral_data':
        all_path2 = func_path(data_folder_name, data_name_no_ligth)
        sub_sd_path = np.load(all_path2.data_path + '_spectraldata.npz')
        file = sub_sd_path
        try:
            spectral_data_no_ligth = file['spectral_data']
        except:    
            spectral_data_no_ligth = file['arr_0']
        file.close()
        
        spectral_data = spectral_data - spectral_data_no_ligth
        print('------ Substraction of spectral data done ------')
    ####################### check if had reco has been done ############
    had_reco_path = all_path.had_reco_path
    had_reco_file = os.path.isfile(had_reco_path)
    if had_reco_file == False or  sub_file == 'spectral_data':
        print('---------- had reconstruction begging ----------')
        #################### Hadamard reconstruction #######################
        patterns = acquisition_parameters.patterns
        Q = wh.walsh2_matrix(Np)
        GT = reconstruction_hadamard(patterns, 'walsh', Q, spectral_data, Np)
        if plot_had_reco == 1:
            plot_reco_without_NN(acquisition_parameters, GT, all_path)
            
    else:
        
        print('--- had_reco exist')
        
        file = np.load(had_reco_path)
        GT = file['arr_0']
        file.close()
        
        GT = np.rot90(GT, 2)
        if plot_had_reco == 1:     
            plt.figure()
            plt.imshow(np.sum(GT, axis = 2))
            plt.colorbar()
            plt.title('GT had reco')
            plt.show()
            
        if substraction == 1 and sub_file == 'GT':
            all_path2 = func_path(data_folder_name, data_name_no_ligth)
            sub_had_reco_path = all_path2.had_reco_path
            file = np.load(sub_had_reco_path)
            GTnl = file['arr_0']
            file.close()
            
            plt.figure()
            plt.imshow(np.sum(GTnl, axis = 2))
            plt.colorbar()
            plt.title('GT no ligth had reco')
            plt.show()
            
            GT = GT - GTnl
            print('------ Substraction of GT done ------')
            
            plt.figure()
            plt.imshow(np.sum(GT, axis = 2))
            plt.colorbar()
            plt.title('GT sub reco')
            plt.show()
            
            # plt.figure()
            # plt.plot(wavelengths, np.mean(np.mean(GT, axis=1), axis=0))
            # plt.title('spectrum sub')
        
        # plot_reco_without_NN(acquisition_parameters, GT, all_path)
                   
    had_reco = 1    
    #%% Neural Network Reconstruction
    if nn_reco == 1:
        nn_reco_file_name = all_path.data_path+'_nn_reco.npz'
        if os.path.isfile(nn_reco_file_name):
            print('--- nn_reco exist')
            file = np.load(nn_reco_file_name)
            reco = file['arr_0']
            file.close()
            
            if substraction == 1:
                sub_nn_reco_path = all_path2.nn_reco_path
                file = np.load(sub_nn_reco_path)
                reco_nl = file['arr_0']
                file.close()
                
                reco = reco - reco_nl
                
            
            rec = np.rot90(reco, 2, axes=(1,0))
            reco = np.rot90(rec, 2, axes=(1,2))
            reco = np.flip(reco, axis=2)
            
            
            
            plt.figure()
            plt.imshow(np.sum(reco, axis = 2))
            plt.title('reco NN reco')
            plt.colorbar()
            
            ####################### spectral view ###################
            size_x = reco.shape[0]
            size_y = reco.shape[1]
            GT50 = reco[round(size_x/4):round(size_x*3/4), round(size_y/4):round(size_y*3/4), :]
            GT25 = reco[round(size_x*3/8):round(size_x*5/8), round(size_y*3/8):round(size_y*5/8), :]
            
            plt.figure()
            plt.plot(acquisition_parameters.wavelengths, np.mean(np.mean(GT25,axis=1),axis=0))
            plt.plot(acquisition_parameters.wavelengths, np.mean(np.mean(GT50,axis=1),axis=0))
            plt.plot(acquisition_parameters.wavelengths, np.mean(np.mean(reco,axis=1),axis=0))
            plt.grid()
            plt.title("% of region from the center of the image nn")
            plt.legend(['25%', '50%', '100%'])
            plt.xlabel(r'$\lambda$ (nm)')
            plt.show()
            
        else:
            if ('spectral_data' in locals()) == False:
                file = np.load(all_path.data_path+'_spectraldata.npz')
                try:
                    spectral_data = file['spectral_data']
                except:
                    spectral_data = file['arr_0']
                    print('npz item : arr_0')
                    
            nn_reco_path = all_path.nn_reco_path
            nn_reco_file = os.path.isfile(nn_reco_path)
            if nn_reco_file == False: 
                print('---------- nn reconstruction begging ----------')
                if Np == 64:
                    img_size_reco = 128
                else:
                    img_size_reco = 64
                                    
                Meas = Np*Np
                network_param = ReconstructionParameters(
                    # Reconstruction network    
                    M = Meas,           # Number of measurements
                    img_size = img_size_reco,      #  Image size of the NN reconstruction
                    arch = 'dc-net',    # Main architecture
                    denoi = 'unet',     # Image domain denoiser (possibility to do not apply, put : None)
                    subs = 'rect',#'var',#,      # Subsampling scheme                    
                    # Training
                    data = 'imagenet',  #'stl10', # Training database
                    N0 = 10,            # Intensity (max of ph./pixel)                    
                    # Optimisation (from train2.py)
                    num_epochs = 30,       # Number of training epochs
                    learning_rate = 0.001, # Learning Rate
                    step_size = 10,        # Scheduler Step Size
                    gamma = 0.5,           # Scheduler Decrease Rate   
                    batch_size = 256,      # Size of the training batch
                    regularization = 1e-7 # Regularisation Parameter
                    )
                
                if network_param.subs == 'rect':
                    cov_folder = 'C:/openspyrit/stat/ILSVRC2012_v10102019/'                    
                elif network_param.subs == 'var':
                    cov_folder = 'C:/openspyrit/stat/ILSVRC2012_v10102019_walsh_randomcrop/'
                
                if network_param.subs == 'var' and network_param.img_size == 64:
                    cov_path = Path(cov_folder) / f'Cov_{network_param.img_size}x{network_param.img_size}.npy'
                else:
                    cov_path = Path(cov_folder) / f'Cov_8_{network_param.img_size}x{network_param.img_size}.npy'
                    
                model_folder = 'C:/openspyrit/models/'
                model, device = setup_reconstruction(cov_path, model_folder, network_param)
                if plot_nn_reco == 1:
                    plot_reco_with_NN(acquisition_parameters, spectral_data, model, device, network_param, all_path, cov_path)    
                    
            else:
                print('--- nn_reco exist')
                if plot_nn_reco == 1:
                    print('---------- nn reconstruction begging ----------')
                    if Np*Np == 4096:
                        Meas = Np*Np-1
                    else:
                        Meas = Np*Np
                    network_param = ReconstructionParameters(
                        # Reconstruction network    
                        M = Meas,#Np*Np-1,        # Number of measurements
                        img_size = 64,      #  Image size of the NN reconstruction
                        arch = 'dc-net',    # Main architecture
                        denoi = 'unet',     # Image domain denoiser (possibility to do not apply, put : None)
                        subs = 'rect',      # Subsampling scheme
                        
                        # Training
                        data = 'imagenet',  # Training database
                        N0 = 10,            # Intensity (max of ph./pixel)
                        
                        # Optimisation (from train2.py)
                        num_epochs = 30,       # Number of training epochs
                        learning_rate = 0.001, # Learning Rate
                        step_size = 10,        # Scheduler Step Size
                        gamma = 0.5,           # Scheduler Decrease Rate   
                        batch_size = 256,      # Size of the training batch
                        regularization = 1e-7 # Regularisation Parameter
                        )
        
                    cov_folder = 'C:/openspyrit/stat/ILSVRC2012_v10102019/'
                    cov_path = Path(cov_folder) / f'Cov_8_{network_param.img_size}x{network_param.img_size}.npy'
                    model_folder = 'C:/openspyrit/models/'
                    model, device = setup_reconstruction(cov_path, model_folder, network_param)
                    plot_reco_with_NN(acquisition_parameters, spectral_data, model, device, network_param, all_path)  
                
    else:
        print('--- nn reco not wished ---')
                                 
#     #%% transfer data to girder
#     if tranfer == 1:
#         transfer_data_2arms(metadata, acquisition_parameters, spectrometer_parameters, DMD_parameters, camPar,
#                                 setup_version, data_folder_name, data_name, collection_acces, upload_metadata=1)
#     #%% Write dataLog in csv file
#     if write_in_csv_file == 1:
#         rows = [
#             {'setup_version': setup_version,
#             'data_folder_name': data_folder_name,
#             'data_name': data_name,
#             'transfered_to_girder': tranfer,
#             'had_reco': had_reco,
#             'nn_reco': nn_reco,
#             'check_data_exist_in_girder': 0,
#             'delete_old_fig': delete_old_fig}
#                 ]      
        
#         with open(csv_file_path, 'a', encoding='UTF8', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             if csv_exist == False:
#                 writer.writeheader()
#             writer.writerows(rows)
    
#     # if inc == 0:
#     #     break
    
# print('total elapsed time = ' + str(round(time.time()-t_tot_0)) + ' s')
#%% mask
if masking == 1:
    white_field_sum = white_field[:,:,800:1200].sum(axis = 2)
    
    plt.figure()
    plt.imshow(white_field_sum)
    plt.title('white field before masking')
    plt.colorbar()
    
mask = np.zeros(white_field_sum.shape)
mask[white_field_sum > 250] = 1

plt.figure()
plt.imshow(mask)
plt.colorbar()
#%% process data by fitting
if process_data == 1:   
        plt.figure()
        plt.plot(wavelengths, np.mean(np.mean(GT, axis=1), axis=0))
        plt.axvline(x = 620, color = 'b', label = 'axvline - full height')
        plt.axvline(x = 634, color = 'r', label = 'axvline - full height')
        plt.grid()
        plt.title('Mean of GT')
            
        plt.figure()
        plt.plot(wavelengths, spectral_data[0, :])
        plt.axvline(x = 620, color = 'b', label = 'axvline - full height')
        plt.axvline(x = 634, color = 'r', label = 'axvline - full height')
        plt.grid()
        plt.title('spectral data for saturation')
        
        # for ii in range(670, 1100):
        #     print([ii, wavelengths[ii]])
        
        plot_fig_fit = 0

        
        # Loop for hadamard and NN reco
        # for inc_nn in range(fit_loop_nbr):
        inc_nn = 1
        if inc_nn == 1:
            if inc_nn == 0:
                size_range = 16                    
                print('Had reco fitting')
            elif inc_nn == 1:
                size_range = 64
                print('NN reco fitting')

            mat = np.empty([size_range, size_range], dtype=float) 
            mat620 = np.empty([size_range, size_range], dtype=float) 
            mat634 = np.empty([size_range, size_range], dtype=float) 
            # Loop on each pixel of the image
            for py in range(size_range):
                print('[py=' + str(py))
                for px in range(size_range):
                    if mask[px, py] == 1:
                        # print('[px=' + str(px) + ', py=' + str(py))
                        P = [px, px, py, py]
                        if inc_nn == 0:
                            Roi = GT[P[0]:P[1]+1, P[2]:P[3]+1,:]
                            titi = 'Had'
                        elif inc_nn == 1:
                            # Pint = np.array(P)*int(64/Np)
                            # Pnn = Pint.tolist()
                            Pnn = [px, px, py, py]
                            Roi = reco[Pnn[0]:Pnn[1]+1, Pnn[2]:Pnn[3]+1,:]
                            titi = 'NN'
        
                        Roivec = np.mean(np.mean(Roi, axis=1), axis=0)
    
                        # Define the Gaussian function
                        def Gauss(x, H, A, x0, sigma):
                            y = H + A*np.exp(-1*(x-x0)**2/(2*sigma**2))
                            return y
                        
                        # Define the 2 Gaussian functions
                        def twoGauss(x, H, A1, mu1, sig1, A2, mu2, sig2):
                            y = H + A1*np.exp(-1*(x-mu1)**2/(2*sig1**2)) + A2*np.exp(-1*(x-mu2)**2/(2*sig2**2))
                            return y
                        
                        # x1 = 670    # => 600nm
                        x1 = 811    # => 617nm
                        x2 = 1092   # => 650nm
                        xdata = wavelengths[x1:x2]
                        ydata = Roivec[x1:x2]
                        # parameters, covariance = curve_fit(Gauss, xdata, ydata, p0=[min(ydata), max(ydata), 630, (x2-x1)/100], bounds=((0, 0, 620, 0), (max(ydata), 2*max(ydata), 640, 10)))
                        try:
                            parameters, covariance = curve_fit(twoGauss, xdata, ydata, p0=[min(ydata), max(ydata), 634, (x2-x1)/100, max(ydata), 625, (x2-x1)/100], 
                                                               bounds=((-40, 0, 629, 0, 0, 615, 0), (max(ydata), 2*max(ydata), 640, 5, 2*max(ydata), 625, 5)))
                            
                            if max(spectral_data[0, :]) == 65535 & 1 == 0:
                                print('!!!!!! Saturation detected !!!!!!')
                                fit_H = 'Saturation'
                                fit_A1 = 'Saturation'
                                fit_mu1 = 'Saturation'
                                fit_sig1 =    'Saturation' 
                                fit_A2 = 'Saturation'
                                fit_mu2 = 'Saturation'
                                fit_sig2 = 'Saturation'
                                sat = 1
                            else:
                                if parameters[0] < 0: parameters[0] = 0
                                if parameters[1] < 0: parameters[1] = 0
                                if parameters[4] < 0: parameters[4] = 0
                                
                                fit_H = parameters[0]
                                fit_A1 = parameters[1]
                                fit_mu1 = parameters[2]
                                fit_sig1 = parameters[3]    
                                fit_A2 = parameters[4]
                                fit_mu2 = parameters[5]
                                fit_sig2 = parameters[6]
                                sat = 0                      
                                
                                fit_y = twoGauss(xdata, fit_H, fit_A1, fit_mu1, fit_sig1, fit_A2, fit_mu2, fit_sig2)
                                fit_y634 = Gauss(xdata, fit_H, fit_A1, fit_mu1, fit_sig1)
                                fit_y620 = Gauss(xdata, fit_H, fit_A2, fit_mu2, fit_sig2)
        
                                mat[px, py] = fit_H#max(fit_y) - fit_H
                                mat620[px,py] = fit_A1# - fit_H#max(fit_y620) - fit_H
                                mat634[px,py] = fit_A2# - fit_H#max(fit_y634) - fit_H
                                 
                        except:
                            print('!!!!! pb with the fit !!!!!!')
                            mat[px, py] = 0
                            mat620[px,py] = 0
                            mat634[px,py] = 0
                    else:
                        mat[px, py] = 0
                        mat620[px,py] = 0
                        mat634[px,py] = 0
                            
            matRatio=mat620/mat634
            matRatio[abs(matRatio)>10] = 0
            
            
            plt.figure()
            plt.imshow(mat)
            plt.title('offset')
            plt.colorbar()
            
            plt.figure()
            plt.imshow(mat620)
            plt.title('carto 620')
            plt.colorbar()
            
            plt.figure()
            plt.imshow(mat634)
            plt.title('carto 634')
            plt.colorbar()
            
            plt.figure()
            plt.imshow(matRatio)
            plt.title('Ratio 620/634')
            plt.colorbar()
            
#%% exploit some points in the image     
from scipy.ndimage import median_filter

ROI = np.zeros((2,2048))

inc_sup = 0
tab_ydata = []
tab_Roivec = []        
maxi = np.amax(mat)

#patient: 60; biopsie: 8
ROI_620 = np.mean(np.mean(reco[25:31, 25:31], axis=1), axis=0)
ROI_634 = np.mean(np.mean(reco[32:34, 32:44], axis=1), axis=0)

#patient: 61; biopsie: 7
ROI_620 = np.mean(np.mean(reco[15:29, 25:39], axis=1), axis=0)
ROI_634 = np.mean(np.mean(reco[41:45, 18:27], axis=1), axis=0)
# ROI_3 = np.mean(np.mean(reco[38:47, 34:40], axis=1), axis=0)

ROI[0, :] = median_filter(ROI_620, size=16, cval=0, mode='constant')
ROI[1, :] = median_filter(ROI_634, size=16, cval=0, mode='constant')
# ROI[2, :] = median_filter(ROI_3, size=16, cval=0, mode='constant')
# x1 = 0
# x2 = 2047
ydata=ydata[x1:x2]
for inc in range(len(ROI)): 
    titi = 'NN'
    
    Roivec = ROI[inc,:]
    ydata = Roivec[x1:x2]
    # parameters, covariance = curve_fit(Gauss, xdata, ydata, p0=[min(ydata), max(ydata), 630, (x2-x1)/100], bounds=((0, 0, 620, 0), (max(ydata), 2*max(ydata), 640, 10)))
    parameters, covariance = curve_fit(twoGauss, xdata, ydata, p0=[min(ydata), max(ydata), 634, (x2-x1)/100, max(ydata), 625, (x2-x1)/100], 
                                       bounds=((-40, 0, 629, 0, 0, 615, 0), (max(ydata), 2*max(ydata), 640, 5, 2*max(ydata), 625, 5)))
    
    if max(spectral_data[0, :]) == 65535 & 1 == 0:
        print('!!!!!! Saturation detected !!!!!!')
        fit_H = 'Saturation'
        fit_A1 = 'Saturation'
        fit_mu1 = 'Saturation'
        fit_sig1 =    'Saturation' 
        fit_A2 = 'Saturation'
        fit_mu2 = 'Saturation'
        fit_sig2 = 'Saturation'
        sat = 1
    else:
        fit_H = parameters[0]
        fit_A1 = parameters[1]
        fit_mu1 = parameters[2]
        fit_sig1 = parameters[3]    
        fit_A2 = parameters[4]
        fit_mu2 = parameters[5]
        fit_sig2 = parameters[6]
        sat = 0                      
        
        fit_y = twoGauss(xdata, fit_H, fit_A1, fit_mu1, fit_sig1, fit_A2, fit_mu2, fit_sig2)
        fit_y634 = Gauss(xdata, fit_H, fit_A1, fit_mu1, fit_sig1)
        fit_y620 = Gauss(xdata, fit_H, fit_A2, fit_mu2, fit_sig2)
    

    # plt.figure()
    # plt.plot(wavelengths, Roivec)
    # plt.axvline(x = 620, color = 'b', label = 'axvline - full height')
    # plt.axvline(x = 634, color = 'r', label = 'axvline - full height')
    # plt.grid()
    # plt.title('in the ROI for')
    
    plt.figure()
    plt.plot(xdata, ydata, 'ob', label='data')
    plt.plot(xdata, fit_y, '-k', label='fit total')
    plt.plot(xdata, fit_y634, '-r', label='fit 634')
    plt.plot(xdata, fit_y620, '-b', label='fit 620')
    plt.grid()
    plt.title('ROI ' + str(inc))
    plt.legend()

#%%


ydata_mean = np.zeros(len(xdata), dtype=float)
spect_mean = np.zeros(len(wavelengths), dtype=float)
for nsup in range(len(tab_ydata)):
    ydata_mean = ydata_mean + tab_ydata[nsup]
    spect_mean = spect_mean + tab_Roivec[nsup]

ydata = ydata_mean/len(tab_ydata)
parameters, covariance = curve_fit(twoGauss, xdata, ydata, p0=[min(ydata), max(ydata), 634, (x2-x1)/100, max(ydata), 625, (x2-x1)/100], 
                                   bounds=((-40, 0, 629, 0, 0, 618, 0), (max(ydata), 2*max(ydata), 640, 5, 2*max(ydata), 630, 5)))

if max(spectral_data[0, :]) == 65535 & 1 == 0:
    print('!!!!!! Saturation detected !!!!!!')
    fit_H = 'Saturation'
    fit_A1 = 'Saturation'
    fit_mu1 = 'Saturation'
    fit_sig1 =    'Saturation' 
    fit_A2 = 'Saturation'
    fit_mu2 = 'Saturation'
    fit_sig2 = 'Saturation'
    sat = 1
else:
    fit_H = parameters[0]
    fit_A1 = parameters[1]
    fit_mu1 = parameters[2]
    fit_sig1 = parameters[3]    
    fit_A2 = parameters[4]
    fit_mu2 = parameters[5]
    fit_sig2 = parameters[6]
    sat = 0                      
    
    fit_y = twoGauss(xdata, fit_H, fit_A1, fit_mu1, fit_sig1, fit_A2, fit_mu2, fit_sig2)
    fit_y634 = Gauss(xdata, fit_H, fit_A1, fit_mu1, fit_sig1)
    fit_y620 = Gauss(xdata, fit_H, fit_A2, fit_mu2, fit_sig2)     
    
    plt.figure()
    plt.plot(wavelengths, spect_mean)
    plt.axvline(x = 620, color = 'b', label = 'axvline - full height')
    plt.axvline(x = 634, color = 'r', label = 'axvline - full height')
    plt.grid()
    plt.title('mean in the ROI for ' + titi + ' reco [px=' + str(px) + ',py=' + str(py) + ']')
    
    plt.figure()
    plt.plot(xdata, ydata, 'ob', label='data')
    plt.plot(xdata, fit_y, '-k', label='fit total')
    plt.plot(xdata, fit_y634, '-r', label='fit 634')
    plt.plot(xdata, fit_y620, '-b', label='fit 620')
    plt.grid()
    plt.title('mean ' + titi + ' reco')
    plt.legend()
    
    Area = np.sum(fit_y - fit_H)
    
    print('Parameters = [', parameters, ' ',Area, ']')
   
if write_result_in_csv_file == 1 & bool(P) == True:
    if sat == 0:
        rows = [
            {'data_folder_name': data_folder_name,
            'data_name': data_name,
            'reco' : titi,
            'offset': np.round(fit_H*10)/10,
            'Amp 1': np.round(fit_A1*10)/10,
            'Lambda 1': np.round(fit_mu1*10)/10,
            'sigma 1': np.round(fit_sig1*10)/10,
            'Amp 2': np.round(fit_A2*10)/10,
            'Lambda 2': np.round(fit_mu2*10)/10,
            'sigma 2': np.round(fit_sig2*10)/10,
            'Area': np.round(Area),
            'image size': Np, 
            'ti': ti}
                ]      
    else:
        rows = [
            {'data_folder_name': data_folder_name,
            'data_name': data_name,
            'reco' : titi,
            'offset': fit_H,
            'Amp 1': fit_A1,
            'Lambda 1': fit_mu1,
            'sigma 1': fit_sig1,
            'Amp 2': fit_A2,
            'Lambda 2': fit_mu2,
            'sigma 2': fit_sig2,
            'Area': Area,
            'image size': Np, 
            'ti': ti}
                ] 
    
    with open(csv_result_file_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result_fieldnames)
        if csv_result_exist == False:
            writer.writeheader()
        writer.writerows(rows)
        

 
 



# if data_folder_name == '2023-06-22_HCL_High_grade_glioma':
#             # |
#             #y|
#             # |
#             # v------->
#             #     x
#     if data_name == 'obj_biopsy_1_center_lesion_source_405nm_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [2, 5, 9, 13] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_biopsy_2_inferior_center_lesion_source_385+405nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [4, 7, 7, 11] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_biopsy_2_inferior_center_lesion_source_385nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [4, 7, 7, 11] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_biopsy_3_center_lesion_source_385+405nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_1000ms_zoom_x1':
#         P = [3, 5, 8, 11] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_biopsy_3bis_2d_loc_source_385nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [2, 6, 7, 11] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_biopsy_3bis_2d_loc_source_405nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [8, 10, 6, 8] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_biopsy_3bis_center_lesion_source_385+405nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [7, 11, 5, 9] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_biopsy_3bis_center_lesion_source_385nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [4, 8, 8, 10] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_biopsy_3bis_center_lesion_source_405nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [7, 10, 4, 9] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_biopsy_4_inferior_deep_limit_lesion_source_385+405nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [3, 7, 6, 11] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_biopsy_5_hyper_flair_source_385+405nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [7, 9, 2, 12] # P = [x1, x2, y1, y2]   
#     if data_name == 'obj_biopsy_6_peri_venticular_source_385+405nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [3, 6, 7, 11] # P = [x1, x2, y1, y2] 
#     if data_name == 'obj_biopsy_7_cortex_inferior_source_385+405nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [9, 11, 6, 8] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_biopsy_10_deep_limit_source_385+405nm_focalised_2_f80mm_lens_Walsh_im_16x16_ti_600ms_zoom_x1':
#         P = [4, 6, 5, 10] # P = [x1, x2, y1, y2]
        
# if data_folder_name == '2023-08-30_HCL_exvivo_LLG':
#     if data_name == 'obj_sample1-interior-portion_source_Laser_405nm_1.2W_A_0.174_and_white_LED_f80mm-P1_Walsh_im_32x32_ti_50ms_zoom_x1':
#         P = [14, 23, 14, 23] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_sample1-interior-portion_source_Laser_405nm_1.2W_A_0.174_and_white_LED_f80mm-P2_Walsh_im_32x32_ti_50ms_zoom_x1':
#         P = [5, 25, 5, 25] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_sample2-anterior-portion_source_Laser_405nm_1.2W_A_0.174_and_white_LED_f80mm-P2_Walsh_im_32x32_ti_50ms_zoom_x1':
#         P = [5, 25, 5, 25] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_sample3-posterior-portion_source_Laser_405nm_1.2W_A_0.174_and_white_LED_f80mm-P2_Walsh_im_32x32_ti_50ms_zoom_x1':
#         P = [5, 27, 5, 25] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_sample4-contrast-enhancement_source_Laser_405nm_1.2W_A_0.174_and_white_LED_f80mm-P2_Walsh_im_32x32_ti_50ms_zoom_x1':
#         P = [5, 27, 5, 25] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_sample4-contrast-enhancement_source_Laser_405nm_1.2W_A_0.174_f80mm-P2_Walsh_im_32x32_ti_50ms_zoom_x1':
#         P = [5, 27, 5, 25] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_sample5-intern-part_source_Laser_405nm_1.2W_A_0.25_f80mm-P2_Walsh_im_32x32_ti_50ms_zoom_x1':
#         P = [14, 25, 15, 27] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_sample5-intern-part_source_Laser_405nm_1.2W_A_0.174_f80mm-P2_Walsh_im_32x32_ti_50ms_zoom_x1':
#         P = [10, 25, 15, 29] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_sample6-anterior-intern-part_source_Laser_405nm_1.2W_A_0.2_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1':
#         P = [2, 13, 6, 14] # P = [x1, x2, y1, y2]    
#     if data_name == 'obj_sample6-anterior-intern-part_source_Laser_405nm_1.2W_A_0.2_f80mm-P2_Walsh_im_32x32_ti_50ms_zoom_x1':
#         P = [4, 21, 13, 30] # P = [x1, x2, y1, y2] 
#     if data_name == 'obj_sample6-anterior-intern-part_source_Laser_405nm_1.2W_A_0.25_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1':
#         P = [1, 11, 4, 13] # P = [x1, x2, y1, y2] 
#     if data_name == 'obj_sample6-anterior-intern-part_source_Laser_405nm_1.2W_A_0.25_f80mm-P2_Walsh_im_32x32_ti_50ms_zoom_x1':
#         P = [4, 21, 13, 30] # P = [x1, x2, y1, y2] 
#     if data_name == 'obj_sample6-anterior-intern-part-bis_source_Laser_405nm_1.2W_A_0.2_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1':
#         P = [2, 13, 8, 15] # P = [x1, x2, y1, y2] 
#     if data_name == 'obj_sample7-anterior-limit_source_Laser_405nm_1.2W_A_0.2_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1':
#         P = [4, 11, 4, 14] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_sample7-anterior-limit_source_Laser_405nm_1.2W_A_0.178_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1':
#         P = [4, 11, 4, 14] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_sample8-lateral-limit_source_Laser_405nm_1.2W_A_0.178_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1':
#         P = [8, 8, 10, 10] # [2, 11, 7, 13] #[4, 8, 10, 14] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_sample8-lateral-limit-bis_source_Laser_405nm_1.2W_A_0.178_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1':
#         P = [2, 11, 7, 13] # P = [x1, x2, y1, y2]
#     if data_name == 'obj_sample9-posterior-limit_source_Laser_405nm_1.2W_A_0.178_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1':
#         P = [5, 11, 5, 11] # P = [x1, x2, y1, y2]    
#     if data_name == 'obj_sample9-posterior-limit-bis_source_Laser_405nm_1.2W_A_0.178_f80mm-P2_Walsh_im_16x16_ti_200ms_zoom_x1':
#         P = [5, 11, 5, 11] # P = [x1, x2, y1, y2] 
