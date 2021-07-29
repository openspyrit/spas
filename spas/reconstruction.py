# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

import numpy as np

def reconstruction_hadamard(patterns: np.ndarray,
                            mode: str,
                            Q: np.ndarray, 
                            M: np.ndarray, 
                            N: int = 64) -> np.ndarray:
    """Reconstruct an image acquired with Hadamard patterns.

    Args:
        patterns (np.ndarray): 
            Array containing an ordered list of the patterns used for 
            acquisition.
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