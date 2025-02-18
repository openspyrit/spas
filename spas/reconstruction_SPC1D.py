# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

import numpy as np
from spas.metadata import AcquisitionParameters

def reconstruction_hadamard(acquisition_parameters: AcquisitionParameters,
                            mode: str,
                            Q: np.ndarray, 
                            M: np.ndarray, 
                            N: int = 64) -> np.ndarray:
    """Reconstruct an image acquired with Hadamard patterns.

    Args:
        acquisition_parameters (AcquisitionParameters):
            Object containing acquisition specifications
        mode (str):
            Select if reconstruction is based on MATLAB, fht or Walsh generated 
            patterns.
        Q (np.ndarray):
            Acquisition matrix used to generate Hadamard patterns.
        M (np.ndarray):
            Spectral data matrix containing acquired spectra.
        N (int, optional): 
            Reconstructed image dimension. Defaults to 64.

    Returns:
        [np.ndarray]: 
            Reconstructed matrix of size NxN pixels.
    """
    
    patterns = acquisition_parameters.patterns
    
    if mode == 'matlab':
        ind_opt = patterns[1::2]
    if mode == 'fht' or mode == 'walsh':
        ind_opt = patterns[0::2]

    ind_opt = np.array(ind_opt)/2

    if mode == 'matlab':
        ind_opt = ind_opt - 1

    ind_opt = ind_opt.astype('int')
    M_breve = M[0::2,:] - M[1::2,:]
    M_Had = np.zeros((N*N, M.shape[1]))
    M_Had[ind_opt,:] = M_breve

    f = np.matmul(Q,M_Had) # Q.T = Q
    frames = np.reshape(f,(N,N,M.shape[1]))
    frames /= N*N
    
    mask_index = acquisition_parameters.mask_index
    if len(mask_index) > 0:
        x_mask_coord = acquisition_parameters.x_mask_coord
        y_mask_coord = acquisition_parameters.y_mask_coord         
        x_mask_length = x_mask_coord[1] - x_mask_coord[0]
        y_mask_length = y_mask_coord[1] - y_mask_coord[0]

        GTnew_vec = np.zeros((x_mask_length*y_mask_length, frames.shape[2]))
        GT_vec = frames.reshape(-1, frames.shape[-1])

        GTnew_vec[mask_index,:] = GT_vec[:len(mask_index),:]
        frames = np.reshape(GTnew_vec, (y_mask_length, x_mask_length, frames.shape[2]))

    return frames


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