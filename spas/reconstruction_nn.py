# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

"""Functions for image reconstruction using Neural Networks.

Implements functions for loading, reconstructing and plotting using Neural 
Networks developed with spyrit. Allows the possibility of "real-time" 
reconstructions and plots using multiprocessing features.
"""

from dataclasses import InitVar, dataclass, field
from time import perf_counter_ns, sleep
from typing import Tuple, Union
from multiprocessing import Queue
import math
import torch
import numpy as np
from matplotlib import pyplot as plt

from spyrit.core.train import load_net   
from spyrit.core.noise import Poisson   
from spyrit.core.meas import HadamSplit   
from spyrit.core.prep import SplitPoisson   
from spyrit.core.recon import PinvNet, DCNet, TikhonovMeasurementPriorDiag    
from spyrit.core.nnet import Unet   
from spyrit.misc.sampling import Permutation_Matrix, reorder
from spyrit.misc.statistics import Cov2Var
from spyrit.misc.walsh_hadamard import walsh2_matrix

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
        cov_path (str): Path to the covariance matrix used for reconstruction.
        It must be a .npy (numpy) or .pt (pytorch) file. It is converted to 
        a torch tensor for reconstruction.
        
        model_folder (str): Folder containing trained models for reconstruction.
        It is unused by the current implementation.
        
        network_params (ReconstructionParameters): Parameters used to load the model.

    Returns:
        Tuple[Union[Pinv_Net, DC2_Net], str]:
            model (Pinv_Net, DC2_Net):
                Loaded model.
            device (str):
                Device to which the model was loaded.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    if cov_path.endswith('.npy'):
        Cov_rec = torch.from_numpy(np.load(cov_path))
    elif cov_path.endswith('.pt'):
        Cov_rec = torch.load(cov_path)
    else:
        raise RuntimeError('Covariance matrix must be a .npy or .pt file')
    
    H =  walsh2_matrix(network_params.img_size)
    
    # Rectangular sampling
    # N.B.: Only for measurements from patterns of size 2**K reconstructed at 
    # size 2**L, with L > K (e.g., measurements are at size 64, reconstructions 
    # at size 128. 
    Ord = torch.zeros(network_params.img_size, network_params.img_size)
    M_xy = math.ceil(network_params.M**0.5)
    Ord[:M_xy, :M_xy] = 1
    
    # Init network     
    Forward = HadamSplit(network_params.M, network_params.img_size, Ord)
    Noise = Poisson(Forward, network_params.N0)
    Prep = SplitPoisson(network_params.N0, Forward)

    if network_params.denoi is None:
        Denoi = torch.nn.Identity()
    else:
        Denoi = Unet()
        
    model = DCNet(Noise, Prep, Cov_rec, Denoi)
    
    # Load trained DC-Net
    net_arch = network_params.arch
    net_denoi = network_params.denoi
    net_data = network_params.data
    # if (network_params.img_size == 128) and (network_params.M == 4096):
    #     net_order   = 'rect'
    # else:
    #     net_order   = 'var'
        
    net_order = network_params.subs
    
    if net_data == 'stl10':
        bs = 1024
    elif net_data == 'imagenet':
        bs = 256
        
    net_suffix  = f'N0_{network_params.N0}_N_{network_params.img_size}_M_{network_params.M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07_light'
    # net_suffix  = f'N0_{network_params.N0}_N_{network_params.img_size}_M_{network_params.M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07'
    # bs = 1024
    # net_suffix  = f'N0_{network_params.N0}_N_{network_params.img_size}_M_{network_params.M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07_seed_0'
    
    net_folder= f'{net_arch}_{net_denoi}_{net_data}/'
    
    net_title = f'{net_arch}_{net_denoi}_{net_data}_{net_order}_{net_suffix}'
    title = 'C:/openspyrit/models/' + net_folder + net_title + '.pth'
    # print(title)
    
    if network_params.denoi is not None:
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
    # print("meas.shape = " + str(meas.shape[0]))
    N_acq = acqui_param.pattern_dimension_x #int((meas.shape[0]/2)**0.5)#
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
    
    # reorder 
    meas = meas.T
    meas = reorder(meas, Perm_acq, Perm_rec)
        
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
    img_size = model.Acq.meas_op.h
    recon = np.zeros((spectral_data.shape[0], img_size, img_size))
    start = perf_counter_ns()

    # model.PreP.set_expe()
    model.prep.set_expe()
    model.to(device)                  
            
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
