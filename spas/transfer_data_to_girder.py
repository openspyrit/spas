# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 14:42:25 2021

@author: mahieu
"""

import girder_client
import os
import shutil
# from singlepixel import read_metadata

def transfer_data(metadata, acquisition_parameters, spectrometer_params, DMD_params, 
                            setup_version, data_folder_name, data_name):
# data_name = 'inverted_telecentric'

    #%%########################## Girder info #################################    
    url = 'https://pilot-warehouse.creatis.insa-lyon.fr/api/v1'
    collectionId = '6140ba6929e3fc10d47dbe3e'# collection_name = 'spc'    
    txt_file = open('C:/private/no_name.txt', 'r', encoding='utf8')
    apiKey = txt_file.read()
    txt_file.close()
    #%%############################## path ####################################
    data_path = '../data/' + data_folder_name + '/' + data_name  # here, data_name is the subfolder
    temp_path = '../temp/data/' + setup_version + '/' + data_folder_name + '/' + data_name
    #%%######################## erase temp folder #############################
    if len(os.listdir('../temp')) != 0:      
        list_TempFolder = os.listdir('../temp')
        for list_temp in list_TempFolder:
            shutil.rmtree('../temp/' + list_temp) 
    #%%################## copy data to temp folder ############################  
    shutil.copytree(data_path, temp_path)
    #%%################### Girder authentification ############################
    gc = girder_client.GirderClient(apiUrl=url)  # Generate the warehouse client
    gc.authenticate(apiKey=apiKey)  # Authentication to the warehouse
    #%%##################### begin data transfer ##############################
    gc.upload('../temp/data/', collectionId, 'collection', reuseExisting=True)
    #%%############## find data folder id to uplaod metada ####################
    girder_data_folder_id = '6149c3ce29e3fc10d47dbffb'
    version_list = gc.listFolder(girder_data_folder_id, 'folder')
    for version_folder in version_list:
        if version_folder['name'] == setup_version:  
            version_folder_id = version_folder['_id']
            data_folder_list = gc.listFolder(version_folder_id, 'folder')
            for temp_data_folder_name in data_folder_list:
                if temp_data_folder_name['name'] == data_folder_name:
                    data_folder_name_id = temp_data_folder_name['_id']                    
                    data_list = gc.listFolder(data_folder_name_id, 'folder')
                    for data_folder in data_list:
                        if data_folder['name'] == data_name:
                            data_folder_id = data_folder['_id']
                            #print('data_folder_id = ' + data_folder_id)                   
    #%%####################### prepare metadata dict ############################
    #metadata, acquisition_parameters, spectrometer_params, DMD_params = read_metadata(data_path + '/' + data_name +'_metadata.json')
    experiment_dict = metadata.__dict__
    acq_par_dict = acquisition_parameters.__dict__
    spectrometer_dict = spectrometer_params.__dict__
    DMD_params_dict = DMD_params.__dict__
    
    experiment_dict2 = {}
    for key in experiment_dict.keys():
        new_key = 'a)_EXP_' + key
        experiment_dict2[new_key] = experiment_dict[key]
        
    acq_par_dict2 = {}
    for key in acq_par_dict.keys():
        new_key = 'b)_ACQ_' + key
        #if not (key == 'patterns' or key == 'measurement_time' or key == 'timestamps' or key == 'wavelengths'):
        acq_par_dict2[new_key] = acq_par_dict[key]   
            #print(key + '= ' + str(acq_par_dict[key]))
    
    spectrometer_dict2 = {}
    for key in spectrometer_dict.keys():
        new_key = 'c)_SPECTRO_' + key
        spectrometer_dict2[new_key] = spectrometer_dict[key]
    
    DMD_params_dict2 = {}
    for key in DMD_params_dict.keys():
        new_key = 'd)_DMD_' + key
        DMD_params_dict2[new_key] = DMD_params_dict[key]
           
    dict = {}
    dict.update(experiment_dict2)
    dict.update(acq_par_dict2)
    dict.update(spectrometer_dict2)
    dict.update(DMD_params_dict2)

    del dict['a)_EXP_output_directory']
    del dict['a)_EXP_pattern_order_source']
    del dict['a)_EXP_pattern_source']
    del dict['a)_EXP_class_description']
    del dict['b)_ACQ_class_description']
    del dict['b)_ACQ_patterns']
    del dict['b)_ACQ_measurement_time']
    del dict['b)_ACQ_timestamps']
    del dict['b)_ACQ_wavelengths']
    del dict['c)_SPECTRO_initial_available_pixels']
    del dict['c)_SPECTRO_store_to_ram'] 
    del dict['c)_SPECTRO_class_description']
    del dict['c)_SPECTRO_detector']#SENS_HAMS11639
    del dict['c)_SPECTRO_firmware_version']#001.011.000.000
    del dict['c)_SPECTRO_fpga_version']#014.000.012.000
    del dict['c)_SPECTRO_dll_version']#9.10.3.0
    del dict['d)_DMD_apps_fpga_temperature']
    del dict['d)_DMD_class_description']
    del dict['d)_DMD_ddc_fpga_temperature']
    del dict['d)_DMD_device_number']         
    del dict['d)_DMD_id']
    del dict['d)_DMD_initial_memory']         
    del dict['d)_DMD_pcb_temperature'] 
    del dict['d)_DMD_bitplanes']  
    del dict['d)_DMD_type'] 
    del dict['d)_DMD_usb_connection']    
    del dict['d)_DMD_ALP_version']  
    #%%################### begin metadata transfer ############################
    gc.addMetadataToFolder(data_folder_id, dict)     
    #%%##################### erase temp folder ################################
    if len(os.listdir('../temp')) != 0:      
        list_TempFolder = os.listdir('../temp')
        for list_temp in list_TempFolder:
            shutil.rmtree('../temp/' + list_temp) 


def transfer_data_2arms(metadata, acquisition_parameters, spectrometer_params, DMD_params, camPar,
                            setup_version, data_folder_name, data_name):

    #unwrap structure into camPar
    camPar.AOI_X = camPar.rectAOI.s32X.value
    camPar.AOI_Y = camPar.rectAOI.s32Y.value
    camPar.AOI_Width = camPar.rectAOI.s32Width.value
    camPar.AOI_Height = camPar.rectAOI.s32Height.value
    #%%########################## Girder info #################################    
    url = 'https://pilot-warehouse.creatis.insa-lyon.fr/api/v1'
    collectionId = '6140ba6929e3fc10d47dbe3e'# collection_name = 'spc'    
    txt_file = open('C:/private/no_name.txt', 'r', encoding='utf8')
    apiKey = txt_file.read()
    txt_file.close()
    #%%############################## path ####################################
    data_path = '../data/' + data_folder_name + '/' + data_name  # here, data_name is the subfolder
    temp_path = '../temp/data/' + setup_version + '/' + data_folder_name + '/' + data_name
    #%%######################## erase temp folder #############################
    if len(os.listdir('../temp')) != 0:      
        list_TempFolder = os.listdir('../temp')
        for list_temp in list_TempFolder:
            shutil.rmtree('../temp/' + list_temp) 
    #%%################## copy data to temp folder ############################  
    shutil.copytree(data_path, temp_path)
    #%%################### Girder authentification ############################
    gc = girder_client.GirderClient(apiUrl=url)  # Generate the warehouse client
    gc.authenticate(apiKey=apiKey)  # Authentication to the warehouse
    #%%##################### begin data transfer ##############################
    gc.upload('../temp/data/', collectionId, 'collection', reuseExisting=True)
    #%%############## find data folder id to uplaod metada ####################
    girder_data_folder_id = '6149c3ce29e3fc10d47dbffb'
    version_list = gc.listFolder(girder_data_folder_id, 'folder')
    for version_folder in version_list:
        if version_folder['name'] == setup_version:  
            version_folder_id = version_folder['_id']
            data_folder_list = gc.listFolder(version_folder_id, 'folder')
            for temp_data_folder_name in data_folder_list:
                if temp_data_folder_name['name'] == data_folder_name:
                    data_folder_name_id = temp_data_folder_name['_id']                    
                    data_list = gc.listFolder(data_folder_name_id, 'folder')
                    for data_folder in data_list:
                        if data_folder['name'] == data_name:
                            data_folder_id = data_folder['_id']
                            #print('data_folder_id = ' + data_folder_id)                   
    #%%####################### prepare metadata dict ############################
    #metadata, acquisition_parameters, spectrometer_params, DMD_params = read_metadata(data_path + '/' + data_name +'_metadata.json')
    experiment_dict = metadata.__dict__
    acq_par_dict = acquisition_parameters.__dict__
    spectrometer_dict = spectrometer_params.__dict__
    DMD_params_dict = DMD_params.__dict__
    CAM_params_dict = camPar.__dict__
    
    experiment_dict2 = {}
    for key in experiment_dict.keys():
        new_key = 'a)_EXP_' + key
        experiment_dict2[new_key] = experiment_dict[key]
        
    acq_par_dict2 = {}
    for key in acq_par_dict.keys():
        new_key = 'b)_ACQ_' + key
        #if not (key == 'patterns' or key == 'measurement_time' or key == 'timestamps' or key == 'wavelengths'):
        acq_par_dict2[new_key] = acq_par_dict[key]   
            #print(key + '= ' + str(acq_par_dict[key]))
    
    spectrometer_dict2 = {}
    for key in spectrometer_dict.keys():
        new_key = 'c)_SPECTRO_' + key
        spectrometer_dict2[new_key] = spectrometer_dict[key]
    
    DMD_params_dict2 = {}
    for key in DMD_params_dict.keys():
        new_key = 'd)_DMD_' + key
        DMD_params_dict2[new_key] = DMD_params_dict[key]
           
    CAM_params_dict2 = {}
    for key in CAM_params_dict.keys():
        if key == 'bandwidth':
            new_key = 'e)_CAM_' + key + ' (MB/s)'
        elif key == 'pixelClock':
            new_key = 'e)_CAM_' + key + ' (MHz)' 
        elif key == 'exposureTime':
            new_key = 'e)_CAM_' + key + ' (ms)'    
        else:
            new_key = 'e)_CAM_' + key
        CAM_params_dict2[new_key] = CAM_params_dict[key]
        
        
    dict = {}
    dict.update(experiment_dict2)
    dict.update(acq_par_dict2)
    dict.update(spectrometer_dict2)
    dict.update(DMD_params_dict2)
    dict.update(CAM_params_dict2)
    
    del dict['a)_EXP_output_directory']
    del dict['a)_EXP_pattern_order_source']
    del dict['a)_EXP_pattern_source']
    del dict['a)_EXP_class_description']
    del dict['b)_ACQ_class_description']
    del dict['b)_ACQ_patterns']
    del dict['b)_ACQ_patterns_wp']
    del dict['b)_ACQ_measurement_time']
    del dict['b)_ACQ_timestamps']
    del dict['b)_ACQ_wavelengths']
    del dict['c)_SPECTRO_initial_available_pixels']
    del dict['c)_SPECTRO_store_to_ram'] 
    del dict['c)_SPECTRO_class_description']
    del dict['c)_SPECTRO_detector']#SENS_HAMS11639
    del dict['c)_SPECTRO_firmware_version']#001.011.000.000
    del dict['c)_SPECTRO_fpga_version']#014.000.012.000
    del dict['c)_SPECTRO_dll_version']#9.10.3.0
    del dict['d)_DMD_apps_fpga_temperature']
    del dict['d)_DMD_class_description']
    del dict['d)_DMD_ddc_fpga_temperature']
    del dict['d)_DMD_device_number']         
    del dict['d)_DMD_id']
    del dict['d)_DMD_initial_memory']         
    del dict['d)_DMD_pcb_temperature'] 
    del dict['d)_DMD_bitplanes']  
    del dict['d)_DMD_type'] 
    del dict['d)_DMD_usb_connection']    
    del dict['d)_DMD_ALP_version']  
    del dict['e)_CAM_hCam']
    del dict['e)_CAM_sInfo']
    del dict['e)_CAM_cInfo']
    del dict['e)_CAM_nBitsPerPixel']
    del dict['e)_CAM_m_nColorMode']
    del dict['e)_CAM_bytes_per_pixel']
    del dict['e)_CAM_rectAOI']
    del dict['e)_CAM_pcImageMemory']
    del dict['e)_CAM_MemID']
    del dict['e)_CAM_pitch']
    del dict['e)_CAM_camActivated']
    del dict['e)_CAM_Exit']
    del dict['e)_CAM_Memory']    
    del dict['e)_CAM_avi']
    del dict['e)_CAM_punFileID']
    del dict['e)_CAM_timeout']
    del dict['e)_CAM_time_array']
    del dict['e)_CAM_black_pattern_num']
    
    #%%################### begin metadata transfer ############################
    gc.addMetadataToFolder(data_folder_id, dict)     
    #%%##################### erase temp folder ################################
    if len(os.listdir('../temp')) != 0:      
        list_TempFolder = os.listdir('../temp')
        for list_temp in list_TempFolder:
            shutil.rmtree('../temp/' + list_temp) 






























