# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

"""Functions for image reconstruction using Neural Networks.

Implements functions for loading, reconstructing and plotting using Neural 
Networks developed with spyrit. Allows the possibility of "real-time" 
reconstructions and plots using multiprocessing features.
"""

from dataclasses import InitVar, dataclass, field
from enum import IntEnum
from time import perf_counter_ns, sleep
from typing import Tuple, Union
from multiprocessing import Queue
from pathlib import Path

import math
import torch
import numpy as np
from matplotlib import pyplot as plt

# from spyrit.core.reconstruction import Pinv_Net, DC2_Net
from spyrit.core.recon import PinvNet, DCNet    # modified by LMW 30/03/2023

# from spyrit.core.training import load_net
from spyrit.core.train import load_net    # modified by LMW 30/03/2023

from spyrit.misc.statistics import Cov2Var
from spyrit.misc.walsh_hadamard import walsh2_matrix

# from spyrit.core.Acquisition import Acquisition_Poisson_approx_Gauss
from spyrit.core.noise import Poisson    # modified by LMW 30/03/2023

# from spyrit.core.Forward_Operator import Forward_operator_Split_ft_had
from spyrit.core.meas import HadamSplit    # modified by LMW 30/03/2023

# from spyrit.core.Preprocess import Preprocess_Split_diag_poisson
from spyrit.core.prep import SplitPoisson    # modified by LMW 30/03/2023
# il y a aussi la classe SplitRowPoisson ???

# from spyrit.core.Data_Consistency import Generalized_Orthogonal_Tikhonov #, Pinv_orthogonal
from spyrit.core.recon import TikhonovMeasurementPriorDiag     # modified by LMW 30/03/2023

# from spyrit.core.neural_network import Unet #, Identity
from spyrit.core.nnet import Unet    # modified by LMW 30/03/2023

from spyrit.misc.sampling import Permutation_Matrix, reorder


from spas.noise import noiseClass
from spas.metadata import AcquisitionParameters

@dataclass
class ReconstructionParameters:
    """Reconstruction parameters for loading a reconstruction model.
    """
    # Reconstruction network
    M: int                  # Number of measurements
    img_size: int           # Image size
    arch: str               # Main architecture
    denoi: str              # Image domain denoiser
    subs: str               # Subsampling scheme
    
    # Training
    data: str               # Training database
    N0: float               # Intensity (max of ph./pixel)
    #sig: float
    
    # Optimisation (from train2.py)
    num_epochs: int         # Number of training Epochs
    learning_rate: float    # Learning Rate
    step_size: int          # Scheduler Step Size
    gamma: float            # Scheduler Decrease Rate   
    batch_size: int         # Size of the training batch
    regularization: float   # Regularisation Parameter
    #checkpoint_model: str  # Optional path to checkpoint model
    #checkpoint_interval: int# Interval between saving model checkpoints
    

def setup_reconstruction(cov_path: str,
                         model_folder: str,
                         network_params: ReconstructionParameters
                         ) -> Tuple[Union[PinvNet,DCNet], str]:    
    """Loads a neural network for reconstruction.

    Limited to measurements from patterns of size 2**K for reconstruction at 
    size 2**L, with L > K (e.g., measurements at size 64, reconstructions at 
    size 128).

    Args:
        cov_path (str): 
            Path to the covariance matrix used for reconstruction.
        model_folder (str): 
            Folder containing trained models for reconstruction.
        network_params (ReconstructionParameters): 
            Parameters used to load the model.

    Returns:
        Tuple[Union[Pinv_Net, DC2_Net], str]:
            model (Pinv_Net, DC2_Net):
                Loaded model.
            device (str):
                Device to which the model was loaded.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    Cov_rec = np.load(cov_path)    
    H =  walsh2_matrix(network_params.img_size)
    
    # Rectangular sampling
    # N.B.: Only for measurements from patterns of size 2**K reconstructed at 
    # size 2**L, with L > K (e.g., measurements are at size 64, reconstructions 
    # at size 128. 
    Ord = np.ones((network_params.img_size, network_params.img_size))
    n_sub = math.ceil(network_params.M**0.5)
    Ord[:,n_sub:] = 0
    Ord[n_sub:,:] = 0
        
    # Init network  
    #Perm_rec = Permutation_Matrix(Ord)
    #Hperm = Perm_rec @ H
    #Pmat = Hperm[:network_params.M,:]

    # init
    # Forward = Forward_operator_Split_ft_had(Pmat, Perm_rec, 
    #                                         network_params.img_size, 
    #                                         network_params.img_size)
    
    Forward = HadamSplit(network_params.M, 
                         network_params.img_size, 
                         Ord)# modified by LMW 30/03/2023
    
    # Noise = Acquisition_Poisson_approx_Gauss(network_params.N0, Forward)
    
    Noise = Poisson(Forward, network_params.N0)
    
    # Prep = Preprocess_Split_diag_poisson(network_params.N0, 
    #                                      network_params.M, 
    #                                      network_params.img_size**2)
    
    Prep = SplitPoisson(network_params.N0, 
                                         network_params.M, 
                                         network_params.img_size**2)
    
    Denoi = Unet()
    # Cov_perm = Perm_rec @ Cov_rec @ Perm_rec.T
    # DC = Generalized_Orthogonal_Tikhonov(sigma_prior = Cov_perm, 
    #                                      M = network_params.M, 
    #                                      N = network_params.img_size**2)
    
    
    
    # model = DC2_Net(Noise, Prep, DC, Denoi)
    model = DCNet(Noise, Prep, Cov_rec, Denoi)

    # # load    
    # net_folder = '{}_{}_{}'.format(
    #     network_params.arch, network_params.denoi, 
    #     network_params.data)
    
    # suffix = '_{}_N0_{}_N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}_reg_{}'.format(
    #     network_params.subs, network_params.N0,   
    #     network_params.img_size, network_params.M, 
    #     network_params.num_epochs, network_params.learning_rate,
    #     network_params.step_size, network_params.gamma,
    #     network_params.batch_size, network_params.regularization)
    
    # torch.cuda.empty_cache() # need to keep this here?
    # title = Path(model_folder) / net_folder / (net_folder + suffix)
    # load_net(title, model, device)
    # model.eval()                    # Mandantory when batchNorm is used 
    # model = model.to(device)
    
    # Load trained DC-Net
    net_arch = network_params.arch
    net_denoi = network_params.denoi
    net_data = network_params.data
    if (network_params.img_size == 128) and (network_params.M == 4096):
        net_order   = 'rect'
    else:
        net_order   = 'var'
    
    bs = 256
    # net_suffix  = f'N0_{network_params.N0}_N_{network_params.img_size}_M_{network_params.M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07_light'
    net_suffix  = f'N0_{network_params.N0}_N_{network_params.img_size}_M_{network_params.M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07_light'
    
    net_folder= f'{net_arch}_{net_denoi}_{net_data}/'
    
    net_title = f'{net_arch}_{net_denoi}_{net_data}_{net_order}_{net_suffix}'
    # title = './model_v2/' + net_folder + net_title
    title = 'C:/openspyrit/models/' + net_folder + net_title
    load_net(title, model, device, False)
    model.eval()                    # Mandantory when batchNorm is used  

    return model, device


def reorder_subsample(meas: np.ndarray,
                      acqui_param: AcquisitionParameters, 
                      recon_param: ReconstructionParameters,
                      recon_cov_path: str = "/path/cov.py",
                      ) -> np.ndarray:
    """Reorder and subsample measurements
    Args:
        meas (np.ndarray):
            Spectral measurements with dimensions (N_wavelength x M_acq), where
            M_acq is the number of acquired patterns
        acqui_param (AcquisitionParameters):
            Parameters used during the acquisition of the spectral measurements
        recon_param (ReconstructionParameters): 
            Parameters of the reconstruction.
        recon_cov_path (str, optional): 
            path to covariance matrix used for reconstruction
    Returns:
        (np.ndarray): 
            Spectral measurements with dimensions (N_wavelength x M_rec), where
            M_rec is the number of patterns considered for reconstruction. 
            Acquisitions can be subsampled a posteriori, leadind to M_rec < M_acq
    """    
    # Dimensions (N.B: images are assumed to be square)
    N_acq = acqui_param.pattern_dimension_x
    N_rec = recon_param.img_size
    N_wav = meas.shape[0]
    
    # Order used for acquisistion
    Ord_acq = -np.array(acqui_param.patterns)[::2]//2   # pattern order
    Ord_acq = np.reshape(Ord_acq, (N_acq,N_acq))        # sampling map
    Perm_acq = Permutation_Matrix(Ord_acq).T
    
    
    # Order used for reconstruction
    if recon_param.subs == 'rect':
        Ord_rec = np.ones((N_rec, N_rec))
        n_sub = math.ceil(recon_param.M**0.5)
        Ord_rec[:,n_sub:] = 0
        Ord_rec[n_sub:,:] = 0
        
    elif recon_param.subs == 'var':
        Cov_rec = np.load(recon_cov_path)
        Ord_rec = Cov2Var(Cov_rec)
    
    Perm_rec = Permutation_Matrix(Ord_rec)
    
    #
    meas = meas.T
    
    # Subsample acquisition permutation matrix (fill with zeros if necessary)
    if N_rec > N_acq:
        
        # Square subsampling in the "natural" order
        Ord_sub = np.zeros((N_rec,N_rec))
        Ord_sub[:N_acq,:N_acq]= -np.arange(-N_acq**2,0).reshape(N_acq,N_acq)
        Perm_sub = Permutation_Matrix(Ord_sub) 
        
        # Natural order measurements (N_acq resolution)
        Perm_raw = np.zeros((2*N_acq**2,2*N_acq**2))
        Perm_raw[::2,::2] = Perm_acq.T     
        Perm_raw[1::2,1::2] = Perm_acq.T
        meas = Perm_raw @ meas
        
        # Zero filling (needed only when reconstruction resolution is higher 
        # than acquisition res)
        zero_filled = np.zeros((2*N_rec**2, N_wav))
        zero_filled[:2*N_acq**2,:] = meas
        
        meas = zero_filled
        
        Perm_raw = np.zeros((2*N_rec**2,2*N_rec**2))
        Perm_raw[::2,::2] = Perm_sub.T     
        Perm_raw[1::2,1::2] = Perm_sub.T
        
        meas = Perm_raw @ meas
        
    elif N_rec == N_acq:
        Perm_sub = Perm_acq[:N_rec**2,:].T
      
    elif N_rec < N_acq:
        # Square subsampling in the "natural" order
        Ord_sub = np.zeros((N_acq,N_acq))
        Ord_sub[:N_rec,:N_rec]= -np.arange(-N_rec**2,0).reshape(N_rec,N_rec)
        Perm_sub = Permutation_Matrix(Ord_sub) 
        Perm_sub = Perm_sub[:N_rec**2,:]
        Perm_sub = Perm_sub @ Perm_acq.T    
        
    #Reorder measurements when reconstruction order is not "natural"  
    if N_rec <= N_acq:   
        # Get both positive and negative coefficients permutated
        Perm = Perm_rec @ Perm_sub
        Perm_raw = np.zeros((2*N_rec**2,2*N_acq**2))
        
    elif N_rec > N_acq:
        Perm = Perm_rec
        Perm_raw = np.zeros((2*N_rec**2,2*N_rec**2))
    
    Perm_raw[::2,::2] = Perm     
    Perm_raw[1::2,1::2] = Perm
    meas = Perm_raw @ meas
    
    return meas[:2*recon_param.M,:].T
        
        
def reconstruct(model: Union[PinvNet, DCNet],
                device: str, 
                spectral_data: np.ndarray, 
                batches : int = 1, 
                #noise_model: noiseClass = None, 
                is_process: bool = False
                ) -> np.ndarray:
    """Reconstructs images from experimental data.

    Using a loaded model, reconstructs images in batches. It can be used in
    'real-time' using multiprocessing to allow reconstruction and acquisition at
    the same time.

    Args:
        model (Union[Pinv_Net,DC2_Net]): 
            Pre-loaded model for reconstruction.
        device (str):
            Device to which the model was loaded. Used to make sure the spectral
            data is loaded to the same device as the model.
        spectral_data (np.ndarray):
            Spectral data acquired, must have the dimensions (spectral 
            dimension x patterns).
        batches (int, optional):
            Number of batches for reconstruction in case the device does not
            have enough memory to reconstruct all spectral channels at once.
            Defaults to 1 (i.e., all spectral channels reconstructed at once).
        is_process (bool, optional): 
            If True, reconstruction is performed in 'real-time' mode, using
            multiprocessing for reconstruction, thus allowing reconstruction and
            acquisition simultaneously. Defaults to False.

    Returns:
        np.ndarray: Reconstructed images with dimensions (spectral dimension x 
        image size x image size).
    """
    
    # noise_model (noiseClass, optional): 
    #     Loaded noise model in case the reconstruction should use a denoising
    #     method. Defaults to None.
    
    proportion = spectral_data.shape[0]//batches # Amount of wavelengths per batch
    
    # img_size = model.Acq.FO.h # image assumed to be square
    # img_size = 128 # Modified by LMW 30/03/2023
    img_size = model.Acq.meas_op.h
    
    recon = np.zeros((spectral_data.shape[0], img_size, img_size))

    start = perf_counter_ns()

    # model.PreP.set_expe()
    model.prep.set_expe() # Modified by LMW 30/03/2023
    model.to(device) # Modified by LMW 30/03/2023                  
            
    with torch.no_grad():
        for batch in range(batches):
                
            lambda_indices = range(proportion * batch, proportion * (batch+1))

            info = (f'batch {batch},'
                f'reconstructed wavelength range: {lambda_indices}')
            
            if is_process:
                info = '#Recon process:' + info   

            print(info)
        
            # C = noise_model.mu[lambda_indices]
            # s = noise_model.sigma[lambda_indices]
            # K = noise_model.K[lambda_indices]
            
            # n = len(C)
                
            # C = torch.from_numpy(C).float().to(device).reshape(n,1,1)
            # s = torch.from_numpy(s).float().to(device).reshape(n,1,1)
            # K = torch.from_numpy(K).float().to(device).reshape(n,1,1)
        
            # M = spectral_data.shape[1]
            
            # torch_img = torch.from_numpy(spectral_data[lambda_indices,:])
            # torch_img = torch_img.float()
            # torch_img = torch.reshape(torch_img, (len(lambda_indices), 1, M)) # batches, channels, patterns
            # torch_img = torch_img.to(device)
            
            spectral_data_torch = torch.tensor(spectral_data[lambda_indices,:],
                                                dtype = torch.float,
                                                device = device)
        
            recon_torch = model.reconstruct_expe(spectral_data_torch)#,
                #len(lambda_indices), 1, model.n, model.n, C, s, K)
                
            recon[lambda_indices,:,:] = recon_torch.cpu().detach().numpy().squeeze()
    
    end = perf_counter_ns()

    time_info = f'Reconstruction time: {(end-start)/1e+6} ms'

    if is_process:
        time_info = '#Recon process:' + time_info

    print(time_info)

    return recon


def reconstruct_process(model: Union[PinvNet, DCNet],
    device: str, queue_to_recon: Queue, queue_reconstructed: Queue, 
    batches: int, noise_model: noiseClass, sleep_time: float = 0.3) -> None:
    """Performs reconstruction in real-time.

    Args:
        model (Union[compNet, noiCompNet, DenoiCompNet]): 
            Pre-loaded model for reconstruction.
        device (str):
            Device to which the model was loaded. Used to make sure the spectral
            data is loaded to the same device as the model.
        queue_to_recon (Queue):
            Multiprocessing queue containing spectral data acquired for
            reconstruction.
        queue_reconstructed (Queue):
            Multiprocessing queue containing reconstructed data for plotting.
            Used by a plotting process to show data in 'real-time'.
        batches (int):
            Number of batches for reconstruction in case the device does not
            have enough memory to reconstruct all data at a single time. E.g.
            when reconstructing data distributed along many wavelengths, there
            may be too many data.
        noise_model (noiseClass):
            Loaded noise model in case the reconstruction should use a denoising
            method. Defaults to None.
        sleep_time (float, optional):
            Estimated time between two reconstructions. If no data is received
            for reconstruction, the process may sleep for this given time in
            seconds. Defaults to 0.3.
    """

    while True:
        if not queue_to_recon.empty():

            message = queue_to_recon.get()

            if isinstance(message, str):
                if message == 'kill':
                    break

            else:
                spectral_data = message
                print('#Recon process: Got data')
                recon_data = reconstruct(model, device, spectral_data, batches,
                                    noise_model, True)
                queue_reconstructed.put(recon_data)

        else:
            sleep(sleep_time)

    # Removing any data left in queue
    while not queue_to_recon.empty():
        queue_to_recon.get_nowait()
    
    queue_reconstructed.put('kill') # Sends a message to stop plotting
    print('#Recon process: Ended reconstruction')


def plot_recon(queue: Queue, sleep_time: float = 0.3) -> None:
    """Plots reconstructed data in 'real-time'.

    Args:
        queue (Queue):
            Multiprocessing queue containing reconstructed data for plotting.
        sleep_time (float, optional): 
            Estimated time between two reconstructions. If no reconstruction is
            received, the plot main loop enters sleep mode. A plot is showed 
            during sleep_time in seconds. Defaults to 0.3.
    """


    print('#Plot process: Entered plot process')
    plt.ion()

    fig = plt.figure()

    while True:
        if not queue.empty():
            message = queue.get()

            if isinstance(message, str):
                if message == 'kill':
                    break
            else:
                recon_data = message
                print('#Plot process: Got plot')                
                plt.clf()
                plt.imshow(np.sum(recon_data, axis=0), cmap='gray')
                plt.title('Sum of wavelengths')
                plt.draw()
                plt.pause(sleep_time)

        else:
            sleep(sleep_time)

    plt.close('all')

    # Removing any data left in queue
    while not queue.empty():
        queue.get_nowait()

    print('#Plot process: Ended plot')