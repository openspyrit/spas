# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins / mahieu'

# import time
# import pickle
import numpy as np
# from spas.acquisition_SPC1D import read_metadata
import spyrit.misc.walsh_hadamard as wh

# from matplotlib import pyplot as plt

def binArray(data, axis, binstep, binsize, func=np.nanmean):
    """
    Binning on an array
    
    Parameters
    ----------
    data : TYPE
        data is your array.
    axis : TYPE
        axis is the axis you want to been.
    binstep : TYPE
        binstep is the number of points between each bin (allow overlapping bins).
    binsize : TYPE
        binsize is the size of each bin.
    func : TYPE, optional
        func is the function you want to apply to the bin (np.max for maxpooling, np.mean for an average ...). The default is np.nanmean.

    Returns
    -------
    data : TYPE
        The binning array.

    """
    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    data = data.transpose(argdims)
    data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data

def live_hadamard_reco(raw_data: np.array, acquisition_params):
    """
    

    Parameters
    ----------
    raw_data : np.array
        DESCRIPTION.
    acquisition_params : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    fisrt_pass = True
    NA = acquisition_params.NAverages
    NLc = len(acquisition_params.Lc)
    NR = acquisition_params.NRepetitions
    Nx = raw_data.shape[0]
    Npatterns = acquisition_params.pattern_amount
    bin_fact = Nx / (Npatterns/2)
    if Npatterns > 1:
        for iNA in range(NA):
            for iLc in range(NLc):
                for iNR in range(NR): 
                    for iNp in range(Npatterns):
                        temp = raw_data[:, :, iNp, iNA, iLc, iNR]
                        # M_sub = temp[:, :, 0::2, :, :, :] - raw_data[:, :, 1::2, :, :, :]
                        if bin_fact != 1:
                            bin_image = binArray(temp, 0, bin_fact, bin_fact)
                        else:
                            bin_image = temp
                            
                        if fisrt_pass == True:
                            spectral_data_all = np.empty((bin_image.shape[0], bin_image.shape[1], Npatterns, NR, NLc, NA), dtype = float)
                            had_reco_all = np.empty((bin_image.shape[0], int(Npatterns/2), bin_image.shape[1], NR, NLc, NA), dtype = float)
                            fisrt_pass = False
                            
                        spectral_data_all[:, :, iNp, iNR, iLc, iNA] = bin_image
                        
                    M_sub = spectral_data_all[:,:,0::2, iNR, iLc, iNA] - spectral_data_all[:,:,1::2, iNR, iLc, iNA]
                    temp_had_reco = wh.fwht(M_sub) / Npatterns
                    had_reco_all[:, :, :, iNR, iLc, iNA] = np.swapaxes(temp_had_reco, 2, 1)  
        
        had_reco_all = np.squeeze(had_reco_all)
        return had_reco_all
    else:
        had_reco_all = None
                

def spatial_reco(acquisition_params, all_path):
    """

    Parameters
    ----------
    acquisition_params : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    arm = 'spatial'
    NA = acquisition_params.NAverages
    NLc = 1#len(acquisition_params.Lc)
    NR = acquisition_params.NRepetitions
    # Npatterns = acquisition_params.pattern_amount
    fisrt_pass = True
    
    for iNA in range(NA):
        for iLc in range(NLc):
            for iNR in range(NR): 
                # for iNp in range(Npatterns):
                try:
                    file_name = arm + '_Ny_' + str(acquisition_params.Nz[iNR]) + 'mm_Gr_' + str(acquisition_params.Lc[iLc][1]) + '_Lc_' + str(acquisition_params.Lc[iLc][0]) + 'nm_NA_' + str(iNA)
                except:
                    file_name = arm + '_Ny_' + str(acquisition_params.Nz) + 'mm_Gr_' + str(acquisition_params.Lc[iLc][1]) + '_Lc_' + str(acquisition_params.Lc[iLc][0]) + 'nm_NA_' + str(iNA)
                    
                path_name = all_path.raw_data_path + '/' + file_name + '.npz'
                raw_data_file = np.load(path_name)
                raw_data = raw_data_file['arr_0']
                raw_data_file.close()
                
                if fisrt_pass == True:
                    spatial_acqui = np.empty((raw_data.T.shape + (NR, NLc, NA)), dtype = np.int16)
                    fisrt_pass = False
                 
                raw_data = np.rot90(raw_data, k=1, axes=(0,1))
                if len(spatial_acqui.shape) == 6:
                    spatial_acqui[:, :, :, iNR, iLc, iNA] = raw_data
                elif len(spatial_acqui.shape) == 5:
                    spatial_acqui[:, :, iNR, iLc, iNA] = raw_data
                else:
                    print('warning, the length of the shape of the spatial acqui is lower than 5')
    
    spatial_acqui = np.squeeze(spatial_acqui)                
    return spatial_acqui

# def hadamard_reco(data_folder_name: str, data_name: str, mean_NA: bool = True, mean_NR: bool = False, save_spectral_data: bool = True, 
#                   save_spatial_data: bool = False, bin_fact: float = 1, zoom: int = 1):
#     """
#     The Hadamard reconstruction for 1D acquisition

#     Parameters
#     ----------
#     data_folder_name : str
#         The folder parent of the data. its name is a type of : '2025-06-17_name'
#     data_name : str
#         The folder of the data. its name is a type of : "obj_USAF_source_white_LED_Walsh_im_128x128_ti_1.0ms_zoom_x1".
#     mean_NA : bool, optional
#         Calculate the mean along the number of averages axis of the hadamard reconstruction matrix. The default is True.
#     mean_NR : bool, optional
#         Calculate the mean along the number of repetitions axis of the hadamard reconstruction matrix. The default is False.
#     save_spectral_data : bool, optional
#         To save the spectral data matrix. The default is True.
#     save_spatial_data : bool, optional
#         To save the spatial data matrix. The default is False.
        
#     Returns
#     -------
#     had_reco_all: np.array
#         The Hadamard reconstruction matrix. Its size is [pattern dim Y, pattern dim X, wavelength, NRepetitions, central wavelength, NAverages]

#     """


#     output_path = '../../data/' + data_folder_name + '/' + data_name
#     saved_DMD_params, saved_spectrograph_params, saved_cam_spat_params, saved_cam_spec_params, saved_acquisition_params = read_metadata(output_path + '/metadata.json')
    
    
#     Npatterns = saved_acquisition_params.pattern_amount
#     snapshot = saved_cam_spat_params.snapshot
#     if snapshot == True:
#         spatial_Npatterns = 1
#     else:
#         spatial_Npatterns = Npatterns
        
#     Nx = saved_cam_spec_params.width
#     Ny = saved_cam_spec_params.height
#     NR = saved_acquisition_params.NRepetitions
#     Lc = saved_acquisition_params.Lc
#     NLc = len(Lc)
#     NA = saved_acquisition_params.NAverages
#     Npx = saved_acquisition_params.pattern_dimension_x
#     Npy = saved_acquisition_params.pattern_dimension_y
#     height = saved_cam_spat_params.height
#     width = saved_cam_spat_params.width  
    
#     # spectral_data_all = np.empty((int(Ny/bin_fact), Nx, Npatterns, NR, NLc, NA), dtype = float)
#     spatial_data_all = np.empty((height, width, 3, spatial_Npatterns, NR, NLc, NA), dtype = float)
#     # bin_image = np.empty((Npy, Nx), dtype = float)
#     # had_reco_all = np.empty((int(Ny/bin_fact), Npx, Nx, NR, NLc, NA), dtype = float)# Ny must will be changed by the wavelength vector
#     t0 = time.time()
#     # save_image = ''#'pickle'
#     fisrt_pass = True
#     for iNA in range(NA):
#         for iLc in range(NLc):
#             for iNR in range(NR): 
#                 for iNp in range(Npatterns):
#                     data_path = output_path + '/raw_data/spectral_NR_' + str(iNR) + '_Gr_' + str(Lc[iLc][1]) + '_Lc_' + str(Lc[iLc][0]) + 'nm_NA_' + str(iNA) + '_NS_' + str(iNp) + '.npz'
                    
#                     file = np.load(data_path)    
#                     npz_image = file['arr_0']
                    
#                     # delete the two fisrt rows
#                     npz_image = np.delete(npz_image, (0), axis=0)
#                     npz_image = np.delete(npz_image, (1), axis=0)
                    
#                     if bin_fact != 1:
#                         bin_image = binArray(npz_image, 0, bin_fact, bin_fact)
#                     else:
#                         bin_image = npz_image
                        
#                     if fisrt_pass == True:
#                         spectral_data_all = np.empty((bin_image.shape[0], bin_image.shape[1], Npatterns, NR, NLc, NA), dtype = float)
#                         had_reco_all = np.empty((bin_image.shape[0], Npx, Nx, NR, NLc, NA), dtype = float)
#                         fisrt_pass = False
                     
#                     spectral_data_all[:, :, iNp, iNR, iLc, iNA] = bin_image

#                     if save_spatial_data:
#                         if snapshot == False:
#                             data_path = output_path + '/raw_data/spatial_NR_' + str(iNR) + '_Gr_' + str(Lc[iLc][1]) + '_Lc_' + str(Lc[iLc][0]) + 'nm_NA_' + str(iNA) + '_NS_' + str(iNp) + '.pkl'
#                             with open(data_path, "rb") as fp:
#                                 pickle_image = pickle.load(fp)
                            
#                             spatial_data_all[:, :, :, iNp, iNR, iLc, iNA] = pickle_image
#                         elif iNp ==1:
#                             data_path = output_path + '/raw_data/spatial_NR_' + str(iNR) + '_Gr_' + str(Lc[iLc][1]) + '_Lc_' + str(Lc[iLc][0]) + 'nm_NA_' + str(iNA) + '_NS_' + str(iNp) + '.pkl'
#                             with open(data_path, "rb") as fp:
#                                 pickle_image = pickle.load(fp)
                            
#                             spatial_data_all[:, :, :, iNp, iNR, iLc, iNA] = pickle_image
                        
                    
#                 M_sub = spectral_data_all[:,:,0::2, iNR, iLc, iNA] - spectral_data_all[:,:,1::2, iNR, iLc, iNA]
#                 had_reco = wh.fwht(M_sub) / Npy
#                 had_reco = np.swapaxes(had_reco, 2, 1)    
#                 had_reco_all[:, :, :, iNR, iLc, iNA] = had_reco
        
#     print(' read raw data, elapsed time = ' + str(time.time() - t0))
    
#     had_reco_all = np.flip(had_reco_all, axis = 0)
    
#     if mean_NA:
#         had_reco_all = np.mean(had_reco_all, axis = 5)
    
#     if mean_NR:
#         had_reco_all = np.mean(had_reco_all, axis = 3)
    
#     t0 = time.time()
#     had_reco_all = np.squeeze(had_reco_all)
    
#     np.savez_compressed(output_path + '/had_reco.npz', had_reco = had_reco_all)
#     print(' save had reco, elapsed time = ' + str(time.time() - t0))
    
    
#     if save_spectral_data:
#         t0 = time.time()
#         spectral_data_all = np.squeeze(spectral_data_all)
#         np.savez_compressed(output_path + '/spectral_data.npz', spectral_data = spectral_data_all)
#         print(' save spectral data, elapsed time = ' + str(time.time() - t0))
        
#     if save_spatial_data:
#         t0 = time.time()
#         spatial_data_all = np.squeeze(spatial_data_all)
        
#         np.savez_compressed(output_path + '/spatial_data.npz', spatial_data = spatial_data_all)
#         print(' save spatial reco, elapsed time = ' + str(time.time() - t0))
    
#     if save_spectral_data:
#         return had_reco_all, spectral_data_all
#     else:
#         return had_reco_all, 0


def reconstruction_raster(M: np.ndarray, N: int = 64) -> np.ndarray:    
    """Reconstruct an image obtained via Raster scan.

    Args:
        M (np.ndarray): 
             Spectral data matrix containing acquired spectra.
        N (int, optional): 
            Reconstructed image dimension. Defaults to 64.

    Returns:
        np.ndarray:
            Reconstructed matrix of size NxN pixels.
    """
    return np.reshape(M,(N,N,M.shape[1]))