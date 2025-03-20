#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 09:08:24 2025

@author: mahieu
"""

from time import perf_counter_ns
from pathlib import Path
from enum import IntEnum
from dataclasses import dataclass, InitVar
from typing import Optional, List, Tuple
from dataclasses_json import dataclass_json
import numpy as np
from tqdm import tqdm

##### DLL for the DMD
try:
    from ALP4 import ALP4, ALP_FIRSTFRAME, ALP_LASTFRAME
    from ALP4 import ALP_AVAIL_MEMORY, ALP_DEV_DYN_SYNCH_OUT1_GATE, tAlpDynSynchOutGate
    # print('ALP4 is ok in Acquisition file')
except:
    class ALP4:
        pass

from spas.metadata_SPC2D import MetaData, AcquisitionParameters

# connect to the DMD
def init_DMD(dmd_lib_version: str = '4.2') -> Tuple[ALP4, int]:
    """Initialize a DMD and clean its allocated memory from a previous use.
    
    Args:
        dmd_lib_version [str]: the version of the DMD library

    Returns:
        Tuple[ALP4, int]: Tuple containing initialized DMD object and DMD
        initial available memory.
    """

    # Initializing DMD
    stop_init = False
    if dmd_lib_version == '4.1':
        print('dmd lib version = ' + dmd_lib_version + ' not installed, please, install it at the location : "openspyrit/spas/alpV41"')
        stop_init = True
    elif dmd_lib_version == '4.2':
        dll_path = Path(__file__).parent.parent.joinpath('lib/alpV42').__str__()
        DMD = ALP4(version='4.2',libDir=dll_path)
    elif dmd_lib_version == '4.3':
        dll_path = Path(__file__).parent.parent.joinpath('lib/alpV43').__str__()
        DMD = ALP4(version='4.3',libDir=dll_path)
    else:
        print('unknown version of dmd library')
        stop_init = True
        
    if stop_init == False:
        DMD.Initialize(DeviceNum=None)
    
        #print(f'DMD initial available memory: {DMD.DevInquire(ALP_AVAIL_MEMORY)}')
        print('DMD connected')
    
        return DMD, DMD.DevInquire(ALP_AVAIL_MEMORY)
    else:
        print('DMD initialisation aborted')
        

# create the class DMDParameters
class DMDTypes(IntEnum):
    """Enumeration of DMD types and respective codes."""
    ALP_DMDTYPE_XGA = 1
    ALP_DMDTYPE_SXGA_PLUS = 2
    ALP_DMDTYPE_1080P_095A = 3
    ALP_DMDTYPE_XGA_07A = 4
    ALP_DMDTYPE_XGA_055A = 5
    ALP_DMDTYPE_XGA_055X = 6
    ALP_DMDTYPE_WUXGA_096A = 7
    ALP_DMDTYPE_WQXGA_400MHZ_090A = 8
    ALP_DMDTYPE_WQXGA_480MHZ_090A = 9
    ALP_DMDTYPE_WXGA_S450 = 12
    ALP_DMDTYPE_DISCONNECT = 255


@dataclass_json
@dataclass
class DMDParameters:
    """Class containing DMD configurations and status.

    Further information: ALP-4.2 API Description (14/04/2020).

    Attributes:
        add_illumination_time_us (int):
            Extra time in microseconds to account for the spectrometer's 
            "dead time".
        initial_memory (int):
            Initial memory available before sending patterns to DMD.
        dark_phase_time_us (int, optional):
            Time in microseconds taken by the DMD mirrors to completely tilt. 
            Minimum time for XGA type DMD is 44 us.
        illumination_time_us (int, optional):
            Duration of the display of one pattern in a DMD sequence. Units in
            microseconds.
        picture_time_us (int, optional):
            Time between the start of two consecutive pictures (i.e. this
            parameter defines the image display rate). Units in microseconds.
        synch_pulse_width_us (int, optional):
            Duration of DMD's frame synch output pulse. Units in microseconds.
        synch_pulse_delay (int, optional):
            Time in microseconds between start of the frame synch output pulse 
            and the start of the pattern display (in master mode).
        device_number (int, optional):
            Serial number of the ALP device.
        ALP_version (int, optional):
            Version number of the ALP device.
        id (int, optional):
            ALP device identifier for a DMD provided by the API.
        synch_polarity (str, optional):
            Frame synch output signal polarity: 'High' or 'Low.'
        trigger_edge (str, optional):
            Trigger input signal slope. Can be a 'Falling' or 'Rising' edge.
        type (str, optional):
            Digital light processing (DLP) chip present in DMD.
        usb_connection (bool, optional):
            True if USB connection is ok.
        ddc_fpga_temperature (float, optional):
            Temperature of the DDC FPGA (IC4) at DMD connection. Units in °C.
        apps_fpga_temperature (float, optional):
            Temperature of the Applications FPGA (IC3) at DMD connection. Units
            in °C.
        pcb_temperature (float, optional):
            Internal temperature of the temperature sensor IC (IC2) at DMD
            connection. Units in °C.
        display_height (int, optional):
            DMD display height in pixels.
        display_width (int, optional):
            DMD display width in pixels.
        patterns (int, optional):
            Number of patterns uploaded to DMD.
        unused_memory (int, optional):
            Memory available after sending patterns to DMD.
        bitplanes (int, optional):
            Bit depth of the patterns to be displayed. Values supported from 1
            to 8.
        DMD (InitVar[ALP4.ALP4], optional):
            Initialization DMD object. Can be used to automatically fill most of
            the DMDParameters' attributes. Unnecessary if reconstructing object
            from JSON file. Defaut is None.
        class_description (str):
            Class description used to improve redability when dumped to JSON
            file. Default is 'DMD parameters'.
    """

    add_illumination_time_us: int
    initial_memory: int

    dark_phase_time_us: Optional[int] = None
    illumination_time_us: Optional[int] = None
    picture_time_us: Optional[int] = None
    synch_pulse_width_us: Optional[int] = None
    synch_pulse_delay: Optional[int] = None

    device_number: Optional[int] = None
    ALP_version: Optional[int] = None
    id: Optional[int] = None

    synch_polarity: Optional[str] = None
    trigger_edge: Optional[str] = None
    
    # synch_polarity_OUT1: Optional[str] = None
    # synch_period_OUT1: Optional[str] = None
    # synch_gate_OUT1: Optional[str] = None
    
    type: Optional[str] = None
    usb_connection: Optional[bool] = None

    ddc_fpga_temperature: Optional[float] = None
    apps_fpga_temperature: Optional[float] = None
    pcb_temperature: Optional[float] = None

    display_height: Optional[int] = None
    display_width: Optional[int] = None

    patterns: Optional[int] = None
    patterns_wp: Optional[int] = None
    unused_memory: Optional[int] = None
    bitplanes: Optional[int] = None
    
    DMD: InitVar[ALP4.ALP4] = None

    class_description: str = 'DMD parameters'


    def __post_init__(self, DMD: Optional[ALP4.ALP4] = None):
        """ Post initialization of attributes.

        Receives a DMD object and directly asks it for its configurations and
        status, then sets the majority of SpectrometerParameters's attributes.
        During reconstruction from JSON, DMD is set to None and the function
        does nothing, letting initialization for the standard __init__ function.

        Args:
            DMD (ALP4.ALP4, optional): 
                Connected DMD. Defaults to None.
        """
        if DMD == None:
            pass

        else:
            self.device_number = DMD.DevInquire(ALP4.ALP_DEVICE_NUMBER)
            self.ALP_version = DMD.DevInquire(ALP4.ALP_VERSION)
            self.id = DMD.ALP_ID.value

            polarity = DMD.DevInquire(ALP4.ALP_SYNCH_POLARITY)
            if polarity == 2006:
                self.synch_polarity = 'High'
            elif polarity == 2007:
                self.synch_polarity = 'Low'

            edge = DMD.DevInquire(ALP4.ALP_TRIGGER_EDGE)
            if edge == 2008:
                self.trigger_edge = 'Falling'
            elif edge == 2009:
                self.trigger_edge = 'Rising'
                
           # synch_polarity_OUT1 = 
                
            self.type = DMDTypes(DMD.DevInquire(ALP4.ALP_DEV_DMDTYPE))

            if DMD.DevInquire(ALP4.ALP_USB_CONNECTION) == 0:
                self.usb_connection = True
            else:
                self.usb_connection = False

            # Temperatures converted to °C
            self.ddc_fpga_temperature = DMD.DevInquire(
                ALP4.ALP_DDC_FPGA_TEMPERATURE)/256
            self.apps_fpga_temperature = DMD.DevInquire(
                ALP4.ALP_APPS_FPGA_TEMPERATURE)/256
            self.pcb_temperature = DMD.DevInquire(
                ALP4.ALP_PCB_TEMPERATURE)/256

            self.display_width = DMD.nSizeX
            self.display_height = DMD.nSizeY

    
    def update_memory(self, unused_memory: int):

        self.unused_memory = unused_memory
        self.patterns = self.initial_memory - unused_memory


    def update_sequence_parameters(self, add_illumination_time, 
                                DMD: Optional[ALP4.ALP4] = None):

        self.bitplanes = DMD.SeqInquire(ALP4.ALP_BITPLANES) 
        self.illumination_time_us = DMD.SeqInquire(ALP4.ALP_ILLUMINATE_TIME)
        self.picture_time_us = DMD.SeqInquire(ALP4.ALP_PICTURE_TIME)
        self.dark_phase_time_us = self.picture_time_us - self.illumination_time_us
        self.synch_pulse_width_us = DMD.SeqInquire(ALP4.ALP_SYNCH_PULSEWIDTH)
        self.synch_pulse_delay = DMD.SeqInquire(ALP4.ALP_SYNCH_DELAY) 
        self.add_illumination_time_us = add_illumination_time



# setup
def calculate_timings(integration_time: float = 1, 
                      integration_delay: int = 0, 
                      add_illumination_time: int = 300, 
                      synch_pulse_delay: int = 0, 
                      dark_phase_time: int = 44,
                      ) -> Tuple[int, int, int]:
    """Calculate spectrometer and DMD dependant timings.

    Args:
        integration_time (float) [ms]: 
            Spectrometer exposure time during one scan in miliseconds. 
            Default is 1 ms.
        integration_delay (int) [µs]:
            Parameter used to start the integration time not immediately after 
            the measurement request (or on an external hardware trigger), but 
            after a specified delay. Unit is based on internal FPGA clock cycle.
            Default is 0 us.
        add_illumination_time (int) [µs]: 
            Extra time in microseconds to account for the spectrometer's 
            "dead time". Default is 365 us.
        synch_pulse_delay (int) [µs]: 
            Time in microseconds between start of the frame synch output pulse 
            and the start of the pattern display (in master mode). Default is
            0 us.
        dark_phase_time (int) [µs]:
            Time in microseconds taken by the DMD mirrors to completely tilt. 
            Minimum time for XGA type DMD is 44 us. Default is 44 us.

    Returns:
        [Tuple]: DMD timings which depend on spectrometer's parameters.
            synch_pulse_width: Duration of DMD's frame synch output pulse. Units
            in microseconds.
            illumination_time: Duration of the display of one pattern in a DMD
            sequence. Units in microseconds.
            picture_time: Time between the start of two consecutive pictures 
            (i.e. this parameter defines the image display rate). Units in
            microseconds.
    """

    illumination_time = (integration_delay/1000 + integration_time*1000 + 
        add_illumination_time)
    picture_time = illumination_time + dark_phase_time
    synch_pulse_width = round(illumination_time/2 + synch_pulse_delay)
    illumination_time = round(illumination_time)
    picture_time = round(picture_time)

    return synch_pulse_width, illumination_time, picture_time


def setup_DMD(DMD: ALP4, 
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


def _sequence_limits(DMD: ALP4, 
                     pattern_compression: int, 
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
    
    import cv2
    
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
  
    t = perf_counter_ns()

    # for adaptative patterns into a ROI
    apply_mask = False
    mask_index = acquisition_params.mask_index
        
    if len(mask_index) > 0:
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
        
        # if pattern_name == 800:
        #     plt.figure()
        #     # plt.imshow(pat_c_re)
        #     # plt.imshow(pat_mask_all_mat)
        #     # plt.imshow(pat_mask_all_mat_DMD)
        #     plt.imshow(np.rot90(patterns,2))
        #     plt.colorbar()
        #     plt.title('pattern n°' + str(pattern_name))
        
        patterns = patterns.ravel()
        
        DMD.SeqPut(
            imgData=patterns.copy(),
            PicOffset=index, 
            PicLoad=1)
    
    print(f'\nTime for sending all patterns: '
          f'{(perf_counter_ns() - t)/1e+9} s')


def setup_patterns(DMD: ALP4, 
                    metadata: MetaData, 
                    DMD_params: DMDParameters, 
                    acquisition_params: AcquisitionParameters,
                    cov_path: str = None, 
                    pattern_to_display: str = 'white', 
                    loop: bool = False) -> None:
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
        loop (bool):
            is to projet in loop, one or few patterns continously (see AlpProjStartCont
            in the doc for more detail). Default is False
    """
    
    file = np.load(Path(metadata.pattern_order_source))
    pattern_order = file['pattern_order']               
    pos_neg = file['pos_neg']
    
    if loop == True:
        pos_neg = False
        if pattern_to_display == 'white':
            pattern_order = np.array(pattern_order[0:1], dtype=np.int16)
        elif pattern_to_display == 'black':
            pattern_order = np.array(pattern_order[1:2], dtype=np.int16)
        elif pattern_to_display == 'gray':
            index = int(np.where(pattern_order == 1953)[0])
            print(index)
            pattern_order = np.array(pattern_order[index:index+1], dtype=np.int16)
        
    bitplanes = 1
    DMD_params.bitplanes = bitplanes

    if (DMD_params.initial_memory - DMD.DevInquire(ALP_AVAIL_MEMORY) == 
        len(pattern_order)) and loop == False:
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
    
    # Confirm memory allocated in DMD
    DMD_params.update_memory(DMD.DevInquire(ALP_AVAIL_MEMORY))


def setup_timings(DMD: ALP4, 
                   DMD_params: DMDParameters, 
                   picture_time: int, 
                   illumination_time: int, 
                   synch_pulse_delay: int, 
                   synch_pulse_width: int, 
                   trigger_in_delay: int,
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


def disconnect_DMD(DMD: ALP4):
    if DMD is not None:       
        # Stop the sequence display
        try:
            DMD.Halt()
            # Free the sequence from the onboard memory (if any is present)
            if (DMD.Seqs):
                DMD.FreeSeq()
    
            DMD.Free()  
            print('DMD disconnected')
            
        except:
            print('probelm to Halt the DMD')   



























