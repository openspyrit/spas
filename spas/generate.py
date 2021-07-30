# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

"""Functions for generating DMD patterns.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from spyrit.misc.statistics import Cov2Var
from spyrit.learning.model_Had_DCAN import Permutation_Matrix
import spyrit.misc.walsh_hadamard as wh


def incrementing_patterns_64(
    save_data: bool = True,
    path: str = '../Patterns/Incrementing_white/',
    prefix: str = 'incrementing') -> np.ndarray:

    """Generates patterns with an increasing number of white columns.

    Args:
        save_data (bool):
            Boolean to decide if patterns should be saved to an output file or
            just returned by the function. Default is True.
        path (str):
            Output folder.
        prefix (str, optional): 
            Output patterns prefix. Defaults to 'incrementing'.
            
    Returns:
        patterns (ndarray):
            3D array containing a list of 768 images of size 768 by 1024 with
            an incrementing number of columns set to 255.
    """
    
    if save_data:
        if not Path(path).exists():
            Path(path).mkdir()

    patterns = np.zeros((768, 768, 1024),dtype=np.uint8)

    for index, pattern in enumerate(patterns):
        for i in range(index):
            pattern[:,128 + i] = 255
        
        patterns[index,:,:] = pattern

        if save_data:
            image = Image.fromarray(pattern)
            image.save(f'{path}{prefix}_{index}.png')
        

    return np.asarray(patterns, dtype=np.uint8)


def hadamard_posneg(H: np.ndarray) -> Tuple[np.ndarray]:
    """Creates positive and negative Hadamard matrices

    Args:
        H (np.ndarray): 
            Hadamard matrix containing all coefficients.

    Returns:
        Tuple[np.ndarray]:
            Tuple containing 2 matrices corresponding to the positive and
            negative Hadamard coefficients.
    """

    Hpos = np.zeros((H.shape[0], H.shape[1]), dtype=np.uint8)
    Hneg = np.zeros((H.shape[0], H.shape[1]), dtype=np.uint8)
    Hpos[np.where(H > 0)] = 2**8 - 1
    Hneg[np.where(H < 0)] = 2**8 - 1

    return Hpos, Hneg


def hadamard_patterns(width: int=1024, height: int=768, N: int=64, N_DMD: int=768, 
                      save_data: bool = False, path: str = None, 
                      prefix: str = None) -> np.ndarray:
    """Generates Hadamard patterns resized for use according with DMD dimensions

    Args:
        width (int):
            DMD width in pixels.
        height (int):
            DMD height in pixels.
        N (int):
            Reconstructed image dimension.
        N_DMD (int):
            Size of the actual area of the DMD where the patterns will be
            created. This area is represented by a square of N_DMD x N_DMD 
            pixels.
        save_data (str):
            Boolean to select if data will be saved to output files. Default is
            False.
        path (str, optional):
            Selects the pattern output folder. By default files will be saved
            into a folder in the current directory. Default is None.
        prefix (str, optional):
            Selects patterns' filename prefix. If prefis is None, patterns will 
            be named 'Hadamard_{N}x{N}_*.png' Default is None.
            
    Returns:
        ndarray:
            3D array containing all the patterns resized to the correct DMD
            dimensions.
    """  
    return resize_to_DMD(width, height, N, N_DMD, 'Hadamard', 'pos_neg', 
                         save_data, path, prefix)

def walsh_patterns(width: int=1024, height: int=768, N: int=64, N_DMD: int=768, 
                      save_data: bool = False, path: str = None, 
                      prefix: str = None) -> np.ndarray:
    """Generates Walsh patterns resized for use according with DMD dimensions

    Args:
        width (int):
            DMD width in pixels.
        height (int):
            DMD height in pixels.
        N (int):
            Reconstructed image dimension.
        N_DMD (int):
            Size of the actual area of the DMD where the patterns will be
            created. This area is represented by a square of N_DMD x N_DMD 
            pixels.
        save_data (str):
            Boolean to select if data will be saved to output files. Default is
            False.
        path (str, optional):
            Selects the pattern output folder. By default files will be saved
            into a folder in the current directory. Default is None.
        prefix (str, optional):
            Selects patterns' filename prefix. If prefis is None, patterns will 
            be named 'Walsh_{N}x{N}_*.png' Default is None.
            
    Returns:
        ndarray:
            3D array containing all the patterns resized to the correct DMD
            dimensions.
    """  
    return resize_to_DMD(width, height, N, N_DMD, 'Walsh', 'pos_neg', 
                         save_data, path, prefix)


def resize_to_DMD(width: int, height: int, N: int, N_DMD: int, 
                  pattern_name: str, method: str, save_data: bool = False, 
                  path: str = None, prefix: str = None) -> np.ndarray:
    """Generates patterns resized for use according with DMD dimensions.

    Args:
        width (int):
            DMD width in pixels.
        height (int):
            DMD height in pixels.
        N (int):
            Reconstructed image dimension.
        N_DMD (int):
            Size of the actual area of the DMD where the patterns will be
            created. This area is represented by a square of N_DMD x N_DMD 
            pixels.
        pattern_name (str):
            Name of the pattern base for creation of patterns, e.g. Hadamard,
            Walsh.
        method (str):
            Creation method. Selects if patterns will be split into positive and
            negative.
        save_data (bool):
            Boolean to select if data will be saved to output files. Default is
            False.
        path (str, optional):
            Selects the pattern output folder. By default files will be saved
            into a folder in the current directory. Default is None.
        prefix (str, optional):
            Selects patterns' filename prefix. Default is None.
    
    Returns:
        DMD_patterns (ndarray):
            3D array containing all the patterns resized to the correct DMD
            dimensions.
    """              
    
    # TO-DO: Add support for other pattern sequences besides Hadamard

    if save_data:
        if path is None:
            path = f'./{pattern_name}_{N}x{N}/'

        if prefix is None:
            prefix = f'{pattern_name}_{N}x{N}_'

        if not Path(path).exists():
            Path(path).mkdir()  
        
    offset_height = (height - N_DMD) // 2
    offset_width = (width - N_DMD) // 2

    # Hadamard patterns using fht
    if pattern_name == 'Hadamard':
        H = Hadamard_Transform_Matrix(N)
        n_patterns = N ** 2
        
    # Walsh-ordered patterns using torch
    if pattern_name == 'Walsh':
        H = wh.walsh2_matrix(N)
        n_patterns = N ** 2

    if method == 'pos_neg':
        Hpos, Hneg = hadamard_posneg(H)
        DMD_patterns = np.zeros((height, width, 2*n_patterns), dtype=np.uint8)
   
    for index in range(n_patterns):
        if method == 'pos_neg':
            
            # Positive patterns have even indexes
            pattern = Image.fromarray(Hpos[index,:].reshape((N,N)))
            pattern = pattern.resize((N_DMD,N_DMD), resample=Image.BOX)
            pattern = np.array(pattern, dtype=np.uint8)
            DMD_patterns[offset_height:height - offset_height, 
                         offset_width:width - offset_width,
                         2*index] = pattern

            # Negative patterns have odd indexes
            pattern = Image.fromarray(Hneg[index,:].reshape((N,N)))
            pattern = pattern.resize((N_DMD,N_DMD), resample=Image.BOX)
            pattern = np.array(pattern, dtype=np.uint8)
            DMD_patterns[offset_height:height - offset_height, 
                         offset_width:width - offset_width,
                         2*index+1] = pattern
        
        if save_data:
            image = Image.fromarray(DMD_patterns[:,:,2*index])
            image.save(f'{path}{prefix}{2*index}.png')

            image = Image.fromarray(DMD_patterns[:,:,2*index+1])
            image.save(f'{path}{prefix}{2*index+1}.png')
        
    return DMD_patterns


def generate_hadamard_order(N: int, name: str, cov_path: str = None, 
                            pos_neg: bool = True) -> None:
    """Generates the Hadamard pattern index order and saves it.

    Args:
        N (int): 
            Hadamard patterns dimension.
        name (str):
            Output file name.
        cov_path (str, optional): 
            Covariance matrix path. Defaults to None.
        pos_neg (bool, optional): 
            Boolean to select if the generated index sequence should consider
            positive and negative patterns. Defaults to True.
    """
    if cov_path is None:
        cov_path = Path(__file__).parent.joinpath(f'../stats/Cov_{N}x{N}.npy')
    
    else:
        cov_path = Path(cov_path)

    cov_path = cov_path.resolve(strict=True)
        
    Cov = np.load(cov_path)
    Var = Cov2Var(Cov)
    P = Permutation_Matrix(Var)
    _,positions = np.where(P == 1)

    if pos_neg:
        # Creating array with positive and negative patterns
        positions *= 2
        positions_  = positions + 1
        pattern_order = np.empty((positions.size + positions_.size,), 
                                dtype=np.uint16)
        pattern_order[0::2] = positions
        pattern_order[1::2] = positions_
    
    else:
        pattern_order = np.asarray(positions, dtype=np.uint16)

    np.savez(cov_path.parent / name, 
        pattern_order=pattern_order, 
        pos_neg=pos_neg)