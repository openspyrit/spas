# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:10:42 2025

@author: chiliaeva

init
piloting
disconnect
"""


from time import sleep, perf_counter_ns
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import NamedTuple, Tuple, List, Optional
from spas.metadata_SPC2D import DMDParameters

from tqdm import tqdm


##### DLL for the DMD
try:
    from ALP4 import ALP4, ALP_FIRSTFRAME, ALP_LASTFRAME, ALP_BITNUM
    from ALP4 import ALP_AVAIL_MEMORY, ALP_DEV_DYN_SYNCH_OUT1_GATE, tAlpDynSynchOutGate
    # print('ALP4 is ok in Acquisition file')
except:
    class ALP4:
        pass
    
    
    

def _init_DMD() -> Tuple[ALP4, int]:
    """Initialize a DMD and clean its allocated memory from a previous use.

    Returns:
        Tuple[ALP4, int]: Tuple containing initialized DMD object and DMD
        initial available memory.
    """

    # Initializing DMD

    dll_path = Path(__file__).parent.parent.joinpath('lib/alpV42').__str__()
    DMD = ALP4(version='4.2',libDir=dll_path)
    # dll_path = Path(__file__).parent.parent.joinpath('lib/alpV43').__str__()
    # DMD = ALP4(version='4.3',libDir=dll_path)
    DMD.Initialize(DeviceNum=None)

    #print(f'DMD initial available memory: {DMD.DevInquire(ALP_AVAIL_MEMORY)}')
    print('DMD connected')

    return DMD, DMD.DevInquire(ALP_AVAIL_MEMORY)


def _setup_DMD(DMD: ALP4, 
              add_illumination_time: int,
              initial_memory: int
              ) -> DMDParameters:
    """Create DMD metadata.

    Creates basic DMD metadata, but leaves most of its fields empty to be set
    later. Sets up the initial free memory present in the DMD.
    This function's name is used to create cohesion between spectrometer and DMD
    related functions.

    Args:
        DMD (ALP4):
            Connected DMD object.
        add_illumination_time (int):
            Extra time in microseconds to account for the spectrometer's 
            "dead time".
        initial_memory (int):
            Initial memory available in DMD after initialization.

    Returns:
        DMDParameters:
            DMD metadata object.
    """

    return DMDParameters(
        add_illumination_time_us=add_illumination_time,
        initial_memory=initial_memory,
        DMD=DMD)


def _sequence_limits(DMD: ALP4, pattern_compression: int, 
                    sequence_lenght: int,
                    pos_neg: bool = True) -> int:
    """Set sequence limits based on a sequence already uploaded to DMD.

    Args:
        DMD (ALP4): 
            Connected DMD object.
        pattern_compression (int):
            Percentage of total available patterns to be present in an
            acquisition sequence.
        sequence_lenght (int):
            Amount of patterns present in DMD memory.
        pos_neg (bool):
            Boolean indicating if sequence is formed by positive and negative
            patterns. Default is True.

    Returns:
        frames (int): 
            Amount of patterns to be used from a sequence based on the pattern
            compression.
    """

    # Choosing beggining of the sequence
    # DMD.SeqControl(ALP_BITNUM, 1)
    DMD.SeqControl(ALP_FIRSTFRAME, 0)

    # Choosing the end of the sequence
    if (round(pattern_compression * sequence_lenght) % 2 == 0) or not (pos_neg):
        frames = round(pattern_compression * sequence_lenght)
    else:
        frames = round(pattern_compression * sequence_lenght) + 1

    DMD.SeqControl(ALP_LASTFRAME, frames - 1)

    return frames


def _update_sequence(DMD: ALP4,
                     DMD_params: DMDParameters,
                     acquisition_params: AcquisitionParameters,
                     pattern_source: str,
                     pattern_prefix: str,
                     pattern_order: List[int],
                     bitplanes: int = 1):
    """Send new complete pattern sequence to DMD.

    Args:
        DMD (ALP4):
            Connected DMD object.
        DMD_params (DMDParameters):
            DMD metadata object to be updated with pattern related data and with
            memory available after patterns are sent to DMD.
        acquisition_params (AcquisitionParameters):
            Acquisition related metadata object. User must partially fill up
            with pattern_compression, pattern_dimension_x, pattern_dimension_y,
            zoom, x and y offest of patterns displayed on the DMD.
        pattern_source (str):
            Pattern source folder.
        pattern_preffix (str):
            Prefix used in pattern naming.
        pattern_order (List[int]):
            List of the pattern indices in a certain order for upload to DMD.
        bitplanes (int, optional): 
            Pattern bitplanes. Defaults to 1.
    """

    path_base = Path(pattern_source)

    seqId = DMD.SeqAlloc(nbImg=len(pattern_order), 
                             bitDepth=bitplanes)
    
    zoom = acquisition_params.zoom
    x_offset = acquisition_params.xw_offset
    y_offset = acquisition_params.yh_offset
    Np = acquisition_params.pattern_dimension_x
           
    dmd_height = DMD_params.display_height
    dmd_width = DMD_params.display_width
    len_im = int(dmd_height / zoom)  

    # print(f'Pattern order size: {len(pattern_order)}')   
    t = perf_counter_ns()

    # for adaptative patterns into a ROI
    apply_mask = False
    mask_index = acquisition_params.mask_index
        
    if mask_index != []:
        apply_mask = True
        Npx = acquisition_params.pattern_dimension_x
        Npy = acquisition_params.pattern_dimension_y
        mask_element_nbr = len(mask_index)
        x_mask_coord = acquisition_params.x_mask_coord
        y_mask_coord = acquisition_params.y_mask_coord 
        x_mask_length = x_mask_coord[1] - x_mask_coord[0]
        y_mask_length = y_mask_coord[1] - y_mask_coord[0]
    
    first_pass = True
    for index,pattern_name in enumerate(tqdm(pattern_order, unit=' patterns', total=len(pattern_order))):
        # read numpy patterns
        path = path_base.joinpath(f'{pattern_prefix}_{pattern_name}.npy')
        im = np.load(path) 
        
        patterns = np.zeros((dmd_height, dmd_width), dtype=np.uint8)
        
        if apply_mask == True: # for adaptative patterns into a ROI 
            pat_mask_all = np.zeros(y_mask_length*x_mask_length) # initialize a vector of lenght = size of the cropped mask           
            pat_mask_all[mask_index] = im[:mask_element_nbr] #pat_re_vec[:mask_element_nbr] # put the pattern into the vector    
            pat_mask_all_mat = np.reshape(pat_mask_all, [y_mask_length, x_mask_length]) # reshape the vector into a matrix of the 2d cropped mask
            # resize the matrix to the DMD size
            pat_mask_all_mat_DMD = cv2.resize(pat_mask_all_mat, (int(dmd_height*x_mask_length/(Npx*zoom)), int(dmd_height*y_mask_length/(Npy*zoom))), interpolation = cv2.INTER_NEAREST)
            
            if first_pass == True:
                first_pass = False
                len_im3 = pat_mask_all_mat_DMD.shape
                    
            patterns[y_offset:y_offset+len_im3[0], x_offset:x_offset+len_im3[1]] = pat_mask_all_mat_DMD 
        else: # send the entire square pattern without the mask
            im_mat = np.reshape(im, [Np,Np])
            im_HD = cv2.resize(im_mat, (int(dmd_height/zoom), int(dmd_height/zoom)), interpolation = cv2.INTER_NEAREST)
            
            if first_pass == True:
                len_im = im_HD.shape
                first_pass = False
                
            patterns[y_offset:y_offset+len_im[0], x_offset:x_offset+len_im[1]] = im_HD  
        
        if pattern_name == 151:
            plt.figure()
            # plt.imshow(pat_c_re)
            # plt.imshow(pat_mask_all_mat)
            # plt.imshow(pat_mask_all_mat_DMD)
            plt.imshow(patterns)
            plt.colorbar()
            plt.title('pattern n°' + str(pattern_name))
        
        patterns = patterns.ravel()
        
        DMD.SeqPut(
            imgData=patterns.copy(),
            PicOffset=index, 
            PicLoad=1)
    
    print(f'\nTime for sending all patterns: '
          f'{(perf_counter_ns() - t)/1e+9} s')


def _update_sequence(DMD: ALP4,
                     DMD_params: DMDParameters,
                     acquisition_params: AcquisitionParameters,
                     pattern_source: str,
                     pattern_prefix: str,
                     pattern_order: List[int],
                     bitplanes: int = 1):
    """Send new complete pattern sequence to DMD.

    Args:
        DMD (ALP4):
            Connected DMD object.
        DMD_params (DMDParameters):
            DMD metadata object to be updated with pattern related data and with
            memory available after patterns are sent to DMD.
        acquisition_params (AcquisitionParameters):
            Acquisition related metadata object. User must partially fill up
            with pattern_compression, pattern_dimension_x, pattern_dimension_y,
            zoom, x and y offest of patterns displayed on the DMD.
        pattern_source (str):
            Pattern source folder.
        pattern_preffix (str):
            Prefix used in pattern naming.
        pattern_order (List[int]):
            List of the pattern indices in a certain order for upload to DMD.
        bitplanes (int, optional): 
            Pattern bitplanes. Defaults to 1.
    """

    path_base = Path(pattern_source)

    seqId = DMD.SeqAlloc(nbImg=len(pattern_order), 
                             bitDepth=bitplanes)
    
    zoom = acquisition_params.zoom
    x_offset = acquisition_params.xw_offset
    y_offset = acquisition_params.yh_offset
    Np = acquisition_params.pattern_dimension_x
           
    dmd_height = DMD_params.display_height
    dmd_width = DMD_params.display_width
    len_im = int(dmd_height / zoom)  

    # print(f'Pattern order size: {len(pattern_order)}')   
    t = perf_counter_ns()

    # for adaptative patterns into a ROI
    apply_mask = False
    mask_index = acquisition_params.mask_index
        
    if mask_index != []:
        apply_mask = True
        Npx = acquisition_params.pattern_dimension_x
        Npy = acquisition_params.pattern_dimension_y
        mask_element_nbr = len(mask_index)
        x_mask_coord = acquisition_params.x_mask_coord
        y_mask_coord = acquisition_params.y_mask_coord 
        x_mask_length = x_mask_coord[1] - x_mask_coord[0]
        y_mask_length = y_mask_coord[1] - y_mask_coord[0]
    
    first_pass = True
    for index,pattern_name in enumerate(tqdm(pattern_order, unit=' patterns', total=len(pattern_order))):
        # read numpy patterns
        path = path_base.joinpath(f'{pattern_prefix}_{pattern_name}.npy')
        im = np.load(path) 
        
        patterns = np.zeros((dmd_height, dmd_width), dtype=np.uint8)
        
        if apply_mask == True: # for adaptative patterns into a ROI 
            pat_mask_all = np.zeros(y_mask_length*x_mask_length) # initialize a vector of lenght = size of the cropped mask           
            pat_mask_all[mask_index] = im[:mask_element_nbr] #pat_re_vec[:mask_element_nbr] # put the pattern into the vector    
            pat_mask_all_mat = np.reshape(pat_mask_all, [y_mask_length, x_mask_length]) # reshape the vector into a matrix of the 2d cropped mask
            # resize the matrix to the DMD size
            pat_mask_all_mat_DMD = cv2.resize(pat_mask_all_mat, (int(dmd_height*x_mask_length/(Npx*zoom)), int(dmd_height*y_mask_length/(Npy*zoom))), interpolation = cv2.INTER_NEAREST)
            
            if first_pass == True:
                first_pass = False
                len_im3 = pat_mask_all_mat_DMD.shape
                    
            patterns[y_offset:y_offset+len_im3[0], x_offset:x_offset+len_im3[1]] = pat_mask_all_mat_DMD 
        else: # send the entire square pattern without the mask
            im_mat = np.reshape(im, [Np,Np])
            im_HD = cv2.resize(im_mat, (int(dmd_height/zoom), int(dmd_height/zoom)), interpolation = cv2.INTER_NEAREST)
            
            if first_pass == True:
                len_im = im_HD.shape
                first_pass = False
                
            patterns[y_offset:y_offset+len_im[0], x_offset:x_offset+len_im[1]] = im_HD  
        
        if pattern_name == 151:
            plt.figure()
            # plt.imshow(pat_c_re)
            # plt.imshow(pat_mask_all_mat)
            # plt.imshow(pat_mask_all_mat_DMD)
            plt.imshow(patterns)
            plt.colorbar()
            plt.title('pattern n°' + str(pattern_name))
        
        patterns = patterns.ravel()
        
        DMD.SeqPut(
            imgData=patterns.copy(),
            PicOffset=index, 
            PicLoad=1)
    
    print(f'\nTime for sending all patterns: '
          f'{(perf_counter_ns() - t)/1e+9} s')


def _setup_patterns_2arms(DMD: ALP4, metadata: MetaData, DMD_params: DMDParameters, 
                   acquisition_params: AcquisitionParameters, camPar: CAM,
                   cov_path: str = None) -> None:
    """Read and send patterns to DMD.

    Reads patterns from a file and sends a percentage of them to the DMD,
    considering positve and negative Hadamard patterns, which should be even in
    number. 
    Prints time taken to read all patterns and send the requested ones
    to DMD. 
    Updates available memory in DMD metadata object (DMD_params).

    Args:
        DMD (ALP4):
            Connected DMD object.
        metadata (MetaData):
            Metadata concerning the experiment, paths, file inputs and file 
            outputs. Must be created and filled up by the user.
        DMD_params (DMDParameters):
            DMD metadata object to be updated with pattern related data and with
            memory available after patterns are sent to DMD.
        acquisition_params (AcquisitionParameters):
            Acquisition related metadata object. User must partially fill up
            with pattern_compression, pattern_dimension_x, pattern_dimension_y,
            zoom, x and y offest of patterns displayed on the DMD.
    """

    file = np.load(Path(metadata.pattern_order_source))
    pattern_order = file['pattern_order']
    pattern_order = pattern_order.astype('int32')
    
    # copy the black pattern image (png) to the number = -1
    # black_pattern_dest_path = Path( metadata.pattern_source + '/' + metadata.pattern_prefix + '_' + '-1.png' )
    black_pattern_dest_path = Path( metadata.pattern_source + '/' + metadata.pattern_prefix + '_' + '-1.npy' )
                                  
    if black_pattern_dest_path.is_file() == False:
        # black_pattern_orig_path = Path( metadata.pattern_source + '/' + metadata.pattern_prefix + '_' + 
        #                               str(camPar.black_pattern_num) + '.png' )
        black_pattern_orig_path = Path( metadata.pattern_source + '/' + metadata.pattern_prefix + '_' + 
                                      str(camPar.black_pattern_num) + '.npy' )
        shutil.copyfile(black_pattern_orig_path, black_pattern_dest_path)                              
        
        
    # add white patterns for the camera
    if camPar.insert_patterns == 1:
        inc = 0
        while True:
            try:
                pattern_order[inc]  # except error from the end of array to stop the loop
                if (inc % camPar.gate_period) == 0:#16) == 0:
                    pattern_order = np.insert(pattern_order, inc, -1) # double white pattern is required if integration time is shorter than 3.85 ms
                    if camPar.int_time_spect < 3.85:
                        pattern_order = np.insert(pattern_order, inc+1, -1)
                        if camPar.int_time_spect < 1.65:
                            pattern_order = np.insert(pattern_order, inc+2, -1)
                            if camPar.int_time_spect < 1:
                                pattern_order = np.insert(pattern_order, inc+1, -1)
                                if camPar.int_time_spect <= 0.6:
                                    pattern_order = np.insert(pattern_order, inc+1, -1)
                inc = inc + 1
            except:
                # print('while loop finished')
                break
       
        # if camPar.int_time_spect < 1.75: # add one pattern at the beginning of the sequence when the integration time of the spectrometer is shorter than 1.75 ms
        #     print('no interleaving')
        #     #pattern_order = np.insert(pattern_order, 0, -1) 
        #     # pattern_order = np.insert(pattern_order, 0, -1)     
        if (len(pattern_order)%2) != 0: # Add one pattern at the end of the sequence if the pattern number is even
            pattern_order = np.insert(pattern_order, len(pattern_order), -1)
            print('pattern order is odd => a black image is automaticly insert, need to be deleted in the case for tuning the spectrometer')
               
    pos_neg = file['pos_neg']

    bitplanes = 1
    DMD_params.bitplanes = bitplanes

    if (DMD_params.initial_memory - DMD.DevInquire(ALP_AVAIL_MEMORY) == 
        len(pattern_order)):
        print('Reusing patterns from previous acquisition')
        acquisition_params.pattern_amount = _sequence_limits(
            DMD, 
            acquisition_params.pattern_compression,
            len(pattern_order),
            pos_neg=pos_neg)

    else:
        if (DMD.Seqs):
            DMD.FreeSeq()

        _update_sequence(DMD, DMD_params, acquisition_params, metadata.pattern_source, metadata.pattern_prefix, 
                         pattern_order, bitplanes)
        print(f'DMD available memory after sequence allocation: '
        f'{DMD.DevInquire(ALP_AVAIL_MEMORY)}')
        acquisition_params.pattern_amount = _sequence_limits(
            DMD, 
            acquisition_params.pattern_compression, 
            len(pattern_order),
            pos_neg=pos_neg)        

    acquisition_params.patterns = (
        pattern_order[0:acquisition_params.pattern_amount])
    
    acquisition_params.patterns_wp = acquisition_params.patterns
    
    # Confirm memory allocated in DMD
    DMD_params.update_memory(DMD.DevInquire(ALP_AVAIL_MEMORY))


def _setup_timings(DMD: ALP4, DMD_params: DMDParameters, picture_time: int, 
                  illumination_time: int, synch_pulse_delay: int, 
                  synch_pulse_width: int, trigger_in_delay: int,
                  add_illumination_time: int) -> None:
    """Setup pattern sequence timings in DMD.

    Send previously user-defined plus calculated timings to DMD.
    Updates DMD metadata with sequence and timing related data.
    This function has no default values for timings and lets the burden of
    setting them to the setup function.

    Args:
        DMD (ALP4):
            Connected DMD object.
        DMD_params (DMDParameters):
            DMD metadata object to be updated with pattern related data and with
            memory available after patterns are sent to DMD.
        picture_time (int):
            Time between the start of two consecutive pictures (i.e. this 
            parameter defines the image display rate). Units in microseconds.
        illumination_time (int):
            Duration of the display of one pattern in a DMD sequence. 
            Units in microseconds.
        synch_pulse_delay (int):
            Time in microseconds between start of the frame synch output pulse 
            and the start of the pattern display (in master mode).
        synch_pulse_width (int):
            Duration of DMD's frame synch output pulse. Units in microseconds.
        trigger_in_delay (int):
            Time in microseconds between the incoming trigger edge and the start
            of the pattern display on DMD (slave mode).
        add_illumination_time (int):
            Extra time in microseconds to account for the spectrometer's 
            "dead time".
    """

    DMD.SetTiming(illuminationTime=illumination_time,
                  pictureTime=picture_time,
                  synchDelay=synch_pulse_delay,
                  synchPulseWidth=synch_pulse_width,
                  triggerInDelay=trigger_in_delay)

    DMD_params.update_sequence_parameters(add_illumination_time, DMD=DMD)
    

def change_patterns(DMD: ALP4, 
                    acquisition_params: AcquisitionParameters, 
                    zoom: int = 1, 
                    xw_offset: int = 0, 
                    yh_offset: int = 0,
                    force_change: bool = False
                    ):
    """
    Delete patterns in the memory of the DMD in the case where the zoom or (x,y) offset change
    
    DMD (ALP4):
        Connected DMD.
    acquisition_params (AcquisitionParameters):
        Acquisition related metadata object. User must partially fill up
        with pattern_compression, pattern_dimension_x, pattern_dimension_y,
        zoom, x and y offest of patterns displayed on the DMD.    
    zoom (int):
        digital zoom. Deafult is x1. 
    xw_offset (int):
        offset int he width direction of the patterns for zoom > 1. 
        Default is 0. 
    yh_offset (int):
        offset int he height direction of the patterns for zoom > 1. 
        Default is 0.
    force_change (bool):
        to force the changement of the pattern sequence. Default is False.
    """
    
    if acquisition_params.zoom != zoom or acquisition_params.xw_offset != xw_offset or acquisition_params.yh_offset != yh_offset or force_change == True:
        if (DMD.Seqs):
            DMD.FreeSeq()









































