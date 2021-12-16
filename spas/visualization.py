# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

import warnings
import colorsys
from typing import Tuple, Optional

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .noise import noiseClass


def spectral_binning(F: np.ndarray, wavelengths: np.ndarray, lambda_min: int, 
    lambda_max: int, n_bin: int, noise: noiseClass=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[noiseClass]]:
    """Bins data over the spectral dimension.

    The output bins are obtained by summing input bins, which garentees 
    that the output bins are Poisson distributed if input the bins are so. 
    Each output bin is the sum of the same number of input bins.
    Bins are calculated using the same amount of pixels each, however not all
    the spectrometer pixels are equally speced, so each bin has its own specific
    wavelength interval.
    Based on the work by N. Ducros, University of Lyon, CREATIS, 03-Jul-2020.
    
    Args:
        F (np.ndarray):
            2D or 3D array containing spectral data. If 2D case, matrix
            dimensions should be WxM, where W is the amount of wavelengths
            acquired and M are the patterns acquired. If 3D case, dimensions
            should be WxNxN, where W is the amount of wavelengths acquired and
            N is the image size for a squared NxN image.
        wavelengths (np.ndarray): 
            1D array containing acquired wavelengths.
        lambda_min (int):
            Minimum wavelength considered for binning.
        lambda_max (int):
            Maximum wavelength considered for binning.
        n_bin (int):
            Number of bins to be calculated.
        noise (noiseClass, optional):
            Noise dataclass containing the dark noise parameters (mu and sigma)
            and the constant K. If informed, it is taken into account for
            binnning.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]: Tuple
        containing data in bins.
            F_bin (np.ndarray):
                2D or 3D array containg n_bin spectral data bins.
            wavelengths_bin (np.ndarray):
                1D array containg n_bin wavelength bins.
            bin_width (np.ndarray):
                Array containing the spectral range per bin, simply put, the
                difference between the maximum and minimum wavelength composing
                a bin. This value depends on the specific bin since the
                wavelengths might not be equally spaced.
            noise_bin (noiseClass, optional):
                If a noiseClass is informed as a function input, noise_bin is
                returned contained a new noiseClass with the convinient values
                of mu, sigma and K according to the generated wavelength bins.
    """

    # Find the indices that fit the spectral range
    lambda_ind, = np.where((wavelengths > lambda_min) & 
                           (wavelengths < lambda_max))
    
    n_tot = len(lambda_ind)
    bin_pixels = n_tot // n_bin
    
    start_ind = (n_tot - bin_pixels*n_bin) // 2
    end_ind = start_ind + bin_pixels*n_bin
    lambda_ind = lambda_ind[start_ind:end_ind]

    wavelengths = wavelengths[lambda_ind]
    

    if noise is not None:
        mu_bin = np.zeros(n_bin)
        sigma_bin = np.zeros(n_bin)
        K_bin = np.zeros(n_bin)

    # Crop across spectral dimension
    # If F is 2D
    if F.ndim == 2:
        F = F[lambda_ind,:]
        is3D = False
        
    # If F is 3D, i.e. the data has been reshaped to an image shape
    elif F.ndim == 3:
        F = F[lambda_ind,:,:]
        [Fx, Fy, Fz] = F.shape
        F = np.reshape(F, (len(lambda_ind),Fy*Fz))
        is3D = True
    
    # Initializing outputs
    F_bin = np.zeros((n_bin,F.shape[1]))
    wavelengths_bin = np.zeros(n_bin)
    bin_width = np.zeros(n_bin)
    
    for b in range(n_bin):
        
        bin_range = range(b * bin_pixels, (b+1) * bin_pixels)
        F_bin[b,:] = np.sum(F[bin_range,:], axis=0)
        wavelengths_bin[b] = np.mean(wavelengths[bin_range])
        bin_width[b] = wavelengths[bin_range[-1]] - wavelengths[bin_range[0]]

        if noise is not None:
            mu_bin[b] = np.sum(noise.mu[bin_range])
            sigma_bin[b] = np.sqrt(np.sum(np.square(noise.sigma[bin_range])))
            K_bin[b] = np.median(noise.K[bin_range])

    if noise is not None:
        noise_bin = noiseClass(mu_bin, sigma_bin, K_bin)

    if is3D:
        F_bin = np.reshape(F_bin, (n_bin,Fy,Fz))

    if noise is None:
        return F_bin, wavelengths_bin, bin_width
    else:
        return F_bin, wavelengths_bin, bin_width, noise_bin
    
def spectral_slicing(F: np.ndarray, wavelengths: np.ndarray, lambda_min: int, 
    lambda_max: int, n_bin: int, noise: noiseClass=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[noiseClass]]:
    """Slices data over the spectral dimension.

    The output are obtained a unique slice, which garentees 
    that the output slices are at the smae position than the spectral binning. 
    Based on the work by N. Ducros, University of Lyon, CREATIS, 03-Jul-2020.
    
    Args:
        F (np.ndarray):
            2D or 3D array containing spectral data. If 2D case, matrix
            dimensions should be WxM, where W is the amount of wavelengths
            acquired and M are the patterns acquired. If 3D case, dimensions
            should be WxNxN, where W is the amount of wavelengths acquired and
            N is the image size for a squared NxN image.
        wavelengths (np.ndarray): 
            1D array containing acquired wavelengths.
        lambda_min (int):
            Minimum wavelength considered for binning.
        lambda_max (int):
            Maximum wavelength considered for binning.
        n_bin (int):
            Number of bins to be calculated.
        noise (noiseClass, optional):
            Noise dataclass containing the dark noise parameters (mu and sigma)
            and the constant K. If informed, it is taken into account for
            binnning.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]: Tuple
        containing data in bins.
            F_bin (np.ndarray):
                2D or 3D array containg n_bin spectral data bins.
            wavelengths_bin (np.ndarray):
                1D array containg n_bin wavelength bins.
            bin_width (np.ndarray):
                Array containing the spectral range per bin, simply put, the
                difference between the maximum and minimum wavelength composing
                a bin. This value depends on the specific bin since the
                wavelengths might not be equally spaced.
            noise_bin (noiseClass, optional):
                If a noiseClass is informed as a function input, noise_bin is
                returned contained a new noiseClass with the convinient values
                of mu, sigma and K according to the generated wavelength bins.
    """
   
    if F.ndim == 3:
        F  = np.transpose(F,(0, 2, 1))

    # Find the indices that fit the spectral range
    lambda_ind, = np.where((wavelengths > lambda_min) & 
                           (wavelengths < lambda_max))
    
    n_tot = len(lambda_ind)
    bin_pixels = n_tot // n_bin
    
    start_ind = (n_tot - bin_pixels*n_bin) // 2
    end_ind = start_ind + bin_pixels*n_bin
    lambda_ind = lambda_ind[start_ind:end_ind]

    wavelengths = wavelengths[lambda_ind]
    

    if noise is not None:
        mu_bin = np.zeros(n_bin)
        sigma_bin = np.zeros(n_bin)
        K_bin = np.zeros(n_bin)

    # Crop across spectral dimension
    # If F is 2D
    if F.ndim == 2:
        F = F[lambda_ind,:]
        is3D = False
        
    # If F is 3D, i.e. the data has been reshaped to an image shape
    elif F.ndim == 3:
        F = F[lambda_ind,:,:]
        [Fx, Fy, Fz] = F.shape
        F = np.reshape(F, (len(lambda_ind),Fy*Fz))
        is3D = True
    
    # Initializing outputs
    F_bin = np.zeros((n_bin,F.shape[1]))
    wavelengths_bin = np.zeros(n_bin)
    bin_width = np.zeros(n_bin)
    
    
    for b in range(n_bin):
        
        bin_range = range(b * bin_pixels, (b+1) * bin_pixels)
        F_bin[b,:] = F[round(np.mean(bin_range)),:]
            
        wavelengths_bin[b] = np.mean(wavelengths[bin_range])
        bin_width[b] = wavelengths[bin_range[-1]] - wavelengths[bin_range[0]]

        if noise is not None:
            mu_bin[b] = np.sum(noise.mu[bin_range])
            # mu_bin[b] = noise.mu[np.mean(bin_range)] ici c'est moi et je ne sais pas si c'est dans le cas où l'on utilisarait un modèle de bruit?????
            sigma_bin[b] = np.sqrt(np.sum(np.square(noise.sigma[bin_range])))
            K_bin[b] = np.median(noise.K[bin_range])

    if noise is not None:
        noise_bin = noiseClass(mu_bin, sigma_bin, K_bin)

    if is3D:
        F_bin = np.reshape(F_bin, (n_bin,Fy,Fz))

    if noise is None:
        return F_bin, wavelengths_bin, bin_width
    else:
        return F_bin, wavelengths_bin, bin_width, noise_bin

def wavelength_to_rgb(wavelength: float,
    gamma: float = 0.8) -> Tuple[float, float, float]:
    """Converts wavelength to RGB.

    Based on https://gist.github.com/friendly/67a7df339aa999e2bcfcfec88311abfc.
    Itself based on code by Dan Bruton: 
    http://www.physics.sfasu.edu/astro/color/spectra.html

    Args:
        wavelength (float): 
            Single wavelength to be converted to RGB.
        gamma (float, optional): 
            Gamma correction. Defaults to 0.8.

    Returns:
        Tuple[float, float, float]:
            RGB value.
    """

    if np.min(wavelength)< 380 or np.max(wavelength) > 750:
        warnings.warn(
            'Some wavelengths are not in the visible range [380-750] nm')

    if (wavelength >= 380 and wavelength <= 440):
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    
    elif (wavelength >= 440 and wavelength <= 490):
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
        
    elif (wavelength >= 490 and wavelength <= 510):
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
        
    elif (wavelength >= 510 and wavelength <= 580):
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
        
    elif (wavelength >= 580 and wavelength <= 645):
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
        
    elif (wavelength >= 645 and wavelength <= 750):
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
        
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    
    return R,G,B


def generate_colormap(wavelength: float, img_size: int,
    gamma: float) -> np.ndarray:
    """Generates colormap for a wavelength.

    Args:
        wavelength (float)):
            Single wavelength used for colormap generation.
        img_size (int):
            Reconstructed image size.
        gamma (float):
            Gamma correction.

    Returns:
        np.ndarray:
            Array with dimensions (img_size,4). Each column corresponds to the
            RGBA values. A stands for alpha or transparency and is currently
            set to 1.
    """

    saturation = np.arange(0,1,1/img_size)
    
    r,g,b = wavelength_to_rgb(wavelength, gamma)
    
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    
    # Creating colormap RGBA (A stands for alpha or transparency)
    colormap = np.ones((img_size,4))
        
    for i in range(img_size):
            
        r,g,b = colorsys.hsv_to_rgb(h, v, saturation[i])
        colormap[i,0] = r
        colormap[i,1] = g
        colormap[i,2] = b
        
    return colormap


def plot_color(F: np.ndarray, wavelengths: np.ndarray,
    gamma: float = 0.8, fontsize: int = 12) -> None:
    """Plots data for each binned wavelength.

    Creates an image in a subplot for each wavelenght in collumns with 4 images
    each for optimized visualization. Each image has its own colorbar and
    colormap.

    Args:
        F (np.ndarray):
            3D Spectral data. First dimension must have n_bin wavelength bins.
            Second and third dimensions contain the reconstructed pixels. 
        wavelengths (np.ndarray): 
            Wavelenghts for plotting (must be already in bins).
        gamma (float, optional): 
            Gamma correction. Defaults to 0.8.
        fontsize (int, optional): 
            Plot fontsize for axes and colorbars. Defaults to 12.
    """

    n_bin = F.shape[0]
    img_size = F.shape[1]
    
    cols = 4
    rows = int(np.round(n_bin/cols))
    
    fig = plt.figure(1, figsize=(16,16), dpi=300)
    
    for bin_ in range(n_bin):
        
        colormap = ListedColormap(
            generate_colormap(wavelengths[bin_], img_size, gamma))
        
        if bin_ + 1 <= rows*cols:
            
            ax = fig.add_subplot(rows, cols, bin_+1)
            ax.axis('off')
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            im = ax.imshow(F[bin_,:,:], cmap=colormap)
            ax.set_title('$\lambda=$'f'{wavelengths[bin_]:.2f}',
                fontsize=fontsize)
            
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(labelsize=fontsize)

            fig.tight_layout()
    
    fig.tight_layout()
    plt.show()