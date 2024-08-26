# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

"""Metadata classes and utilities.

Metadata classes to keep and save all relevant data during an acquisition.
Utility functions to recreate objects from JSON files, save them to JSON and to
improve readability.
"""

import json  
from datetime import datetime
from enum import IntEnum
from dataclasses import dataclass, InitVar, field
from typing import Optional, Union, List, Tuple, Optional
from pathlib import Path
import os
from dataclasses_json import dataclass_json
import numpy as np
import ctypes as ct
import pickle
##### DLL for the DMD
try:
    import ALP4
except: # in the cas the DLL of the DMD is not installed
    class ALP4:
        pass
    setattr(ALP4, 'ALP4',  None)
    print('DLL of the DMD not installed')
##### DLL for the spectrometer Avantes 
try:
    from msl.equipment.resources.avantes import MeasConfigType
except: # in the cas the DLL of the spectrometer is not installed
    class MeasConfigType:
        pass
    MeasConfigType =  None
    print('DLL of the spectrometer not installed !!!')

##### DLL for the camera
try:
    from pyueye import ueye
    dll_pyueye_installed = 1
except:
    dll_pyueye_installed = 0
    print('DLL of the cam not installed !!')

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
class MetaData:
    """ Class containing overall acquisition parameters and description.

    Metadata concerning the experiment, paths, file inputs and file outputs.
    This class is adapted to be reconstructed from a JSON file.

    Attributes:
        output_directory (Union[str, Path], optional):
            Directory where multiple related acquisitions will be stored.
        pattern_order_source (Union[str, Path], optional):
            File where the order of patterns to be sent to DMD is specified. It
            can be a text file containing a list of pattern indeces or a numpy
            file containing a covariance matrix from which the pattern order is
            calculated.
        pattern_source (Union[str, Path], optional):
            Pattern source folder.
        pattern_prefix (str):
            Prefix used in pattern naming.
        experiment_name (str):
            Prefix of all files related to a single acquisition. Files will
            appear with the following string pattern: 
            experiment_name + '_' + filename.
        light_source (str):
            Light source used to illuminate an object during acquisition.
        object (str):
            Object imaged during acquisition.
        filter (str):
            Light filter used.
        description (str):
            Acqusition experiment description.
        date (str, optional):
            Acquisition date. Automatically set when object is created. Default
            is None.
        time (str, optional):
            Time when metadata object is created. Set automatically by
            __post_init__(). Default is None.
        class_description (str):
            Class description used to improve redability when dumped to JSON
            file. Default is 'Metadata'.
    """

    pattern_prefix: str
    experiment_name: str

    light_source: str
    object: str
    filter: str
    description: str

    output_directory: Union[str, Path]
    pattern_order_source: Union[str, Path]
    pattern_source: Union[str, Path]

    date: Optional[str] = None
    time: Optional[str] = None

    class_description: str = 'Metadata'

    def __post_init__(self):
        """Sets time and date of object cretion and deals with paths"""

        today = datetime.today()
        self.date = '--/--/----' #today.strftime('%d/%m/%Y')            
        self.time = '--:--:--' #today.strftime('%I:%M:%S %p')

        # If parameter is str, turn it into Path
        if isinstance(self.output_directory, str):
            self.output_directory = Path(self.output_directory)

        # If parameter is Path or was turned into a Path, resolve it and get the
        # str format.
        if issubclass(self.output_directory.__class__, Path):
            self.output_directory = str(self.output_directory.resolve())


        if isinstance(self.pattern_order_source, str):
            self.pattern_order_source = Path(self.pattern_order_source)

        if issubclass(self.pattern_order_source.__class__, Path):
            self.pattern_order_source = str(
                self.pattern_order_source.resolve())


        if isinstance(self.pattern_source, str):
            self.pattern_source = Path(self.pattern_source)

        if issubclass(self.pattern_source.__class__, Path):
            self.pattern_source = str(self.pattern_source.resolve())


@dataclass_json
@dataclass
class CAM:
    """Class containing IDS camera configurations.

    Further information: https://en.ids-imaging.com/manuals/ids-software-suite/ueye-manual/4.95/en/c_programmierung.html.

    Attributes: 
        hCam (ueye.c_uint): Handle of the camera.
        sInfo (ueye.SENSORINFO):sensor information : [SensorID [c_ushort] = 566;
                                                      strSensorName [c_char_Array_32] = b'UI388xCP-M';
                                                      nColorMode [c_char] = b'\x01';
                                                      nMaxWidth [c_uint] = 3088;
                                                      nMaxHeight [c_uint] = 2076;
                                                      bMasterGain [c_int] = 1;
                                                      bRGain [c_int] = 0;
                                                      bGGain [c_int] = 0;
                                                      bBGain [c_int] = 0;
                                                      bGlobShutter [c_int] = 0;
                                                      wPixelSize [c_ushort] = 240;
                                                      nUpperLeftBayerPixel [c_char] = b'\x00';
                                                      Reserved].
        cInfo (ueye.BOARDINFO):Camera information: [SerNo [c_char_Array_12] = b'4103219888';
                                                    ID [c_char_Array_20] = b'IDS GmbH';
                                                    Version [c_char_Array_10] = b'';
                                                    Date [c_char_Array_12] = b'30.11.2017';
                                                    Select [c_ubyte] = 1;
                                                    Type [c_ubyte] = 100;
                                                    Reserved [c_char_Array_8] = b'';]                
        nBitsPerPixel (ueye.c_int): number of bits per pixel (8 for monochrome, 24 for color).
        m_nColorMode (ueye.c_int): color mode : Y8/RGB16/RGB24/REG32.
        bytes_per_pixel (int): bytes_per_pixel = int(nBitsPerPixel / 8).
        rectAOI (ueye.IS_RECT()): rectangle of the Area Of Interest (AOI):  s32X [c_int] = 0;
                                                                            s32Y [c_int] = 0;
                                                                            s32Width [c_int] = 3088;
                                                                            s32Height [c_int] = 2076;                
        pcImageMemory (ueye.c_mem_p()): memory allocation.
        MemID (ueye.int()): memory identifier.
        pitch (ueye.INT()): ???.
        fps (float): set frame per second.
        gain (int): Set gain between [0 - 100].
        gainBoost (str): Activate gain boosting ("ON") or deactivate ("OFF").
        gamma (float): Set Gamma between [1 - 2.5] to change the image contrast
        exposureTime (float): Set the exposure time between [0.032 - 56.221]
        blackLevel (int): Set the black level between [0 - 255] to set an offset in the image. It is adviced to put 5 for noise measurement
        camActivated (bool) : need to to know if the camera is ready to acquire (1: yes, 0: No) 
        pixelClock (int) : the pixel clock, three values possible : [118, 237, 474] (MHz)
        bandwidth (float) the bandwidth (in MByte/s) is an approximate value which is calculated based on the pixel clock
        Memory (bool) : a boolean to know if the memory inside the camera is busy [1] or free [0]
        Exit (int) : if Exit = 2 => excute is_ExitCamera function (disables the hCam camera handle and releases the memory) | if Exit = 0 => allow to init cam, after that, Exit = 1
        vidFormat (str) : save video in the format avi or bin (for binary)
        gate_period (int) : a second TTL is sent by the DMD to trigg the camera, and based on the fisrt TTL to trigg the spectrometer. camera trigger period = gate_period*(spectrometer trigger period)
        trigger_mode (str) : hard or soft
        avi (ueye.int) : A pointer that returns the instance ID which is needed for calling the other uEye AVI functions
        punFileID (ueye.c_int) : a pointer in which the instance ID is returned. This ID is needed for calling other functions.
        timeout (int) : a time which stop the camera that waiting for a TTL
        time_array (List[float]) : the time array saved after each frame received on the camera
        int_time_spect (float) : is egal to the integration time of the spectrometer, it is need to know this value because of the rolling shutter of the monochrome IDS camera
        black_pattern_num (int) : is number inside the image name of the black pattern (for the hyperspectral arm, or white pattern for the camera arm) to be inserted betweem the Hadamard patterns
        insert_patterns (int) : 0 => no insertion / 1=> insert white patterns for the camera
        acq_mode (str) : mode of the acquisition => 'video' or 'snapshot' mode
    """
    
    hCam: Optional[ueye.c_uint] = None
    sInfo: Optional[ueye.SENSORINFO] = None
    cInfo: Optional[ueye.BOARDINFO] = None
    nBitsPerPixel: Optional[ueye.c_int] = None
    m_nColorMode: Optional[ueye.c_int] = None
    bytes_per_pixel: Optional[int] = None
    rectAOI: Optional[ueye.IS_RECT] = None
    pcImageMemory: Optional[ueye.c_mem_p] = None
    MemID: Optional[ueye.c_int] = None
    pitch: Optional[ueye.c_int] = None
    fps: Optional[float] = None
    gain: Optional[int] = None
    gainBoost: Optional[str] = None
    gamma: Optional[float] = None
    exposureTime: Optional[float] = None
    blackLevel: Optional[int] = None
    camActivated : Optional[bool] = None
    pixelClock : Optional[int] = None
    bandwidth : Optional[float] = None
    Memory : Optional[bool] = None
    Exit : Optional[int] = None
    vidFormat : Optional[str] = None
    gate_period : Optional[int] = None
    trigger_mode : Optional[str] = None
    avi : Optional[ueye.int] = None
    punFileID : Optional[ueye.c_int] = None
    timeout : Optional[int] = None
    time_array : Optional[Union[List[float], str]] = field(default=None, repr=False)
    int_time_spect : Optional[float] = None
    black_pattern_num : Optional[int] = None
    insert_patterns : Optional[int] = None
    acq_mode : Optional[str] = None
                   
    class_description: str = 'IDS camera configuration'
    
    def undo_readable_class_CAM(self) -> None:
        """Changes the time_array attribute from `str` to `List` of `int`."""
        
        def to_float(str_arr):
            arr = []
            for s in str_arr:
                try:
                    num = float(s)
                    arr.append(num)
                except ValueError:
                    pass
            return arr
        
        if self.time_array:
            self.time_array = (
                self.time_array.strip('[').strip(']').split(', '))
            self.time_array = to_float(self.time_array)
            self.time_array = np.asarray(self.time_array)
    
    @staticmethod
    def readable_class_CAM(cam_params_dict: dict) -> dict:
        # pass
        """Turns list of time_array into a string.
        convert the c_type structure (sInfo, cInfo and rectAOI) into a nested dict
        change the bytes type item into str
        change the c_types item into their value
        """

        readable_cam_dict = {}
        readable_cam_dict_temp = cam_params_dict#camPar.to_dict()#
        inc = 0
        for item in readable_cam_dict_temp:
            stri = str(type(readable_cam_dict_temp[item]))
            # print('----- item : ' + item)
            if item == 'sInfo' or item == 'cInfo' or item == 'rectAOI':
                readable_cam_dict[item] = dict()
                try:
                    for sub_item in readable_cam_dict_temp[item]._fields_:
                        new_item = item + '-' + sub_item[0]
                        try:
                            att = getattr(readable_cam_dict_temp[item], sub_item[0]).value
                        except:
                            att = getattr(readable_cam_dict_temp[item], sub_item[0])
                        
                        if type(att) == bytes:
                            att = str(att)
                            
                        readable_cam_dict[item][sub_item[0]] = att
                except:
                    try:
                        for sub_item in readable_cam_dict_temp[item]:
                            # print('----- sub_item : ' + sub_item)
                            new_item = item + '-' + sub_item
                            att = readable_cam_dict_temp[item][sub_item]
                            
                            if type(att) == bytes:
                                att = str(att)
                                
                            readable_cam_dict[item][sub_item] = att
                    except:
                        print('warning, impossible to read the subitem of readable_cam_dict_temp[item]')                        
                        
            elif stri.find('pyueye') >=0:
                try:
                    readable_cam_dict[item] = readable_cam_dict_temp[item].value
                except:
                    readable_cam_dict[item] = readable_cam_dict_temp[item]
            elif item == 'time_array':
                readable_cam_dict[item] = str(readable_cam_dict_temp[item])
            else:
                readable_cam_dict[item] = readable_cam_dict_temp[item]
                            
        return readable_cam_dict
    

@dataclass_json
@dataclass
class AcquisitionParameters:
    """Class containing acquisition specifications and timing results.

    This class is adapted to be reconstructed from a JSON file.

    Attributes:
        pattern_compression (float):
            Percentage of total available patterns to be present in an
            acquisition sequence.
        pattern_dimension_x (int):
            Length of reconstructed image that defines pattern length.
        pattern_dimension_y (int):
            Width of reconstructed image that defines pattern width.
        zoom (int):
            numerical zoom of the patterns
        xw_offset (int):
            offset of the pattern in the DMD for zoom > 1 in the width (x) direction
        yh_offset (int):
            offset of the pattern in the DMD for zoom > 1 in the heihgt (y) direction    
        pattern_amount (int, optional):
            Quantity of patterns sent to DMD for an acquisition. This value is
            calculated by an external function. Default in None.
        acquired_spectra (int, optional):
            Amount of spectra actually read from the spectrometer. This value is
            calculated by an external function. Default in None.
        mean_callback_acquisition_time_ms (float, optional):
            Mean time between 2 callback executions during an acquisition. This 
            value is calculated by an external function. Default in None.
        total_callback_acquisition_time_s (float, optional):
            Total time of callback executions during an acquisition. This value
            is calculated by an external function. Default in None.
        mean_spectrometer_acquisition_time_ms (float, optional):
            Mean time between 2 spectrometer measurements during an acquisition
            based on its own internal clock. This value is calculated by an
            external function. Default in None.
        total_spectrometer_acquisition_time_s (float, optional):
            Total time of spectrometer measurements during an acquisition
            based on its own internal clock. This value is calculated by an
            external function. Default in None.
        saturation_detected (bool, optional):
            Boolean incating if saturation was detected during acquisition.
            Default is None.
        patterns (Union[List[int],str], optional) = None
            List `int` or `str` containing all patterns sent to the DMD for an
            acquisition sequence. This value is set by an external function and
            its type can be modified by multiple functions during object
            creation, manipulation, when dumping to a JSON file or
            when reconstructing an AcquisitionParameters object from a JSON
            file. It is intended to be of type List[int] most of the execution
            List[int]time. Default is None.
        wavelengths (Union[np.ndarray, str], optional):
            Array of `float` type corresponding to the wavelengths associated
            with spectrometer's start and stop pixels.
        timestamps (Union[List[float], str], optional):
            List of `float` type elapsed time between each measurement
            made by the spectrometer based on its internal clock. Units in 
            milliseconds. Default is None.
        measurement_time (Union[List[float], str], optional):
            List of `float` type elapsed times between each callback. Units in
            milliseconds. Default is None.
        class_description (str):
            Class description used to improve redability when dumped to JSON
            file. Default is 'Acquisition parameters'.
    """

    pattern_compression: float
    pattern_dimension_x: int
    pattern_dimension_y: int
    zoom: int
    xw_offset: int
    yh_offset: int
    
    pattern_amount: Optional[int] = None
    acquired_spectra: Optional[int] = None

    mean_callback_acquisition_time_ms: Optional[float] = None
    total_callback_acquisition_time_s: Optional[float] = None
    mean_spectrometer_acquisition_time_ms: Optional[float] = None
    total_spectrometer_acquisition_time_s: Optional[float] = None

    saturation_detected: Optional[bool] = None

    patterns: Optional[Union[List[int], str]] = field(default=None, repr=False)
    patterns_wp: Optional[Union[List[int], str]] = field(default=None, repr=False)
    wavelengths: Optional[Union[np.ndarray, str]] = field(default=None, 
                                                        repr=False)
    timestamps: Optional[Union[List[float], str]] = field(default=None, 
                                                        repr=False)
    measurement_time: Optional[Union[List[float], str]] = field(default=None,
                                                            repr=False)

    class_description: str = 'Acquisition parameters'


    def undo_readable_pattern_order(self) -> None:
        """Changes the patterns attribute from `str` to `List` of `int`.

        When reconstructing an AcquisitionParameters object from a JSON file,
        this method turns the patterns, wavelengths, timestamps and 
        measurement_time attributes from a string to a list of integers
        containing the pattern indices used in that acquisition.
        """
        
        def to_float(str_arr):
            arr = []
            for s in str_arr:
                try:
                    num = float(s)
                    arr.append(num)
                except ValueError:
                    pass
            return arr
        
        self.patterns = self.patterns.strip('[').strip(']').split(', ')
        self.patterns = [int(s) for s in self.patterns if s.isdigit()]
        try:
            self.patterns_wp = self.patterns_wp.text.strip('[').strip(']').split(', ')
            self.patterns_wp = [int(s) for s in self.patterns_wp if s.isdigit()]
        except:
            print('patterns_wp has no attribute ''strip''')    

        if self.wavelengths:
            self.wavelengths = (
                self.wavelengths.strip('[').strip(']').split(', '))
            self.wavelengths = to_float(self.wavelengths)
            self.wavelengths = np.asarray(self.wavelengths)

        else:
            print('wavelenghts not present in metadata.'
            ' Reading data in legacy mode.')

        if self.timestamps:
            self.timestamps = self.timestamps.strip('[').strip(']').split(', ')
            self.timestamps = to_float(self.timestamps)
        else:
            print('timestamps not present in metadata.'
            ' Reading data in legacy mode.')

        if self.measurement_time:
            self.measurement_time = (
                self.measurement_time.strip('[').strip(']').split(', '))
            self.measurement_time = to_float(self.measurement_time)
        else:
            print('measurement_time not present in metadata.'
            ' Reading data in legacy mode.')


    @staticmethod
    def readable_pattern_order(acquisition_params_dict: dict) -> dict:
        """Turns list of patterns into a string.

        Turns the list of pattern attributes from an AcquisitionParameters 
        object (turned into a dictionary) into a string that will improve
        readability once all metadata is dumped into a JSON file.
        This function must be called before dumping.

        Args:
            acquisition_params_dict (dict): Dictionary obtained from converting 
            an AcquisitionParameters object.

        Returns:
            [dict]: Modified dictionary with acquisition parameters metadata.
        """

        def _hard_coded_conversion(data):
            s = '['
            for value in data:
                s += f'{value:.4f}, '
            s = s[:-2]
            s += ']'

            return s

        readable_dict = acquisition_params_dict
        readable_dict['patterns'] = str(readable_dict['patterns'])
        readable_dict['patterns_wp'] = str(readable_dict['patterns_wp'])
        
        readable_dict['wavelengths'] = _hard_coded_conversion(
            readable_dict['wavelengths'])
    
        readable_dict['timestamps'] = _hard_coded_conversion(
            readable_dict['timestamps'])

        readable_dict['measurement_time'] = _hard_coded_conversion(
            readable_dict['measurement_time'])

        return readable_dict


    def update_timings(self, timestamps: np.ndarray, 
                       measurement_time: np.ndarray):
        """Updates acquisition timings.

        Args:
            timestamps (ndarray): 
                Array of `float` type elapsed time between each measurement made
                by the spectrometer based on its internal clock. Units in 
                milliseconds.
            measurement_time (ndarray):
                Array of `float` type elapsed times between each callback. Units
                in milliseconds.
        """
        self.mean_callback_acquisition_time_ms = np.mean(measurement_time)
        self.total_callback_acquisition_time_s = np.sum(measurement_time) / 1000
        self.mean_spectrometer_acquisition_time_ms = np.mean(
            timestamps, dtype=np.float64)
        self.total_spectrometer_acquisition_time_s = np.sum(timestamps) / 1000

        self.timestamps = timestamps
        self.measurement_time = measurement_time



@dataclass_json
@dataclass
class SpectrometerParameters:
    """Class containing spectrometer configurations.

    Further information: AvaSpec Library Manual (Version 9.10.2.0).

    Attributes:
        high_resolution (bool):
            True if 16-bit AD Converter is used. False if 14-bit ADC is used.
        initial_available_pixels (int):
            Number of pixels available in spectrometer.
        detector (str):
            Name of the light detector.
        firmware_version (str, optional):
            Spectrometer firmware version.
        dll_version (str, optional):
            Spectrometer dll version.
        fpga_version (str, optional):
            Internal FPGA version.
        integration_delay_ns (int, optional):
            Parameter used to start the integration time not immediately after 
            the measurement request (or on an external hardware trigger), but 
            after a specified delay. Unit is based on internal FPGA clock cycle.
        integration_time_ms (float, optional):
            Spectrometer exposure time during one scan in miliseconds.
        start_pixel (int, optional):
            Initial pixel data received from spectrometer.
        stop_pixel (int, optional):
            Last pixel data received from spectrometer.
        averages (int, optional):
            Number of averages in a single measurement.
        dark_correction_enable (bool, optional):
            Enable dynamic dark current correction.
        dark_correction_forget_percentage (int, optional):
            Percentage of the new dark value pixels that has to be used. e.g., 
            a percentage of 100 means only new dark values are used. A 
            percentage of 10 means that 10 percent of the new dark values is
            used and 90 percent of the old values is used for drift correction.
        smooth_pixels (int, optional):
            Number of neighbor pixels used for smoothing, max. has to be smaller
            than half the selected pixel range because both the pixels on the
            left and on the right are used.
        smooth_model (int, optional):
            Smoothing model. Currently a single model is supported in which the
            spectral data is averaged over a number of pixels on the detector 
            array. For example, if the smoothpix parameter is set to 2, the
            spectral data for all pixels x(n) on the detector array will be
            averaged with their neighbor pixels x(n-2), x(n-1), x(n+1) and
            x(n+2).
        saturation_detection (bool, optional):
            Enable detection of saturation/overexposition in pixels.
        trigger_mode (int, optional):
            Trigger mode (0 = Software, 1 = Hardware, 2 = Single Scan).
        trigger_source (int, optional):
            Trigger source (0 = external trigger, 1 = sync input).
        trigger_source_type (int, optional):
            Trigger source type (0 = edge trigger, 1 = level trigger).
        store_to_ram (int, optional):
            Define how many scans can be stored in RAM. In DynamicRAM mode, can
            be set to 0 to indicate infinite measurements.
        configs: InitVar[MeasConfigType]:
            Initialization object containing data to create SpectrometerData
            object. Unnecessary if reconstructing object from JSON file Defaut
            is None.
        version_info: InitVar[Tuple[str]]:
            Initialization variable used for receiving firmware, dll and FPGA
            version data. Unnecessary if reconstructing object from JSON file.
        class_description (str):
            Class description used to improve redability when dumped to JSON
            file. Default is 'Spectrometer parameters'.
    """

    high_resolution: bool
    initial_available_pixels: int
    detector: str
    firmware_version: Optional[str] = None
    dll_version: Optional[str] = None
    fpga_version: Optional[str] = None

    integration_delay_ns: Optional[int] = None
    integration_time_ms: Optional[float] = None

    start_pixel: Optional[int] = None
    stop_pixel: Optional[int] = None
    averages: Optional[int] = None

    dark_correction_enable: Optional[bool] = None
    dark_correction_forget_percentage: Optional[int] = None

    smooth_pixels: Optional[int] = None
    smooth_model: Optional[int] = None

    saturation_detection: Optional[bool] = None

    trigger_mode: Optional[int] = None
    trigger_source: Optional[int] = None
    trigger_source_type: Optional[int] = None

    store_to_ram: Optional[int] = None

    configs: InitVar[MeasConfigType] = None
    version_info: InitVar[Tuple[str]] = None

    class_description: str = 'Spectrometer parameters'


    def __post_init__(self, configs: Optional[MeasConfigType] = None,
                      version_info: Optional[Tuple[str, str, str]] = None):
        """Post initialization of attributes.

        Receives the data sent to spectrometer and some version data and unwraps
        everything to set the majority of SpectrometerParameters's attributes.
        During reconstruction from JSON, arguments of type InitVar (configs and
        version_info) are set to None and the function does nothing, letting
        initialization for the standard __init__ function.

        Args:
            configs (MeasConfigType, optional):
                Object containing configurations sent to spectrometer.
                Defaults to None.
            version_info (Tuple[str, str, str], optional):
                Tuple containing firmware, dll and FPGA version data. Defaults
                to None.
        """
        if configs is None or version_info is None:
            pass

        else:
            self.fpga_version, self.firmware_version, self.dll_version = (
                version_info)
            
            self.integration_delay_ns = configs.m_IntegrationDelay
            self.integration_time_ms = configs.m_IntegrationTime

            self.start_pixel = configs.m_StartPixel
            self.stop_pixel = configs.m_StopPixel
            self.averages = configs.m_NrAverages

            self.dark_correction_enable = configs.m_CorDynDark.m_Enable
            self.dark_correction_forget_percentage = (
                configs.m_CorDynDark.m_ForgetPercentage)

            self.smooth_pixels = configs.m_Smoothing.m_SmoothPix
            self.smooth_model = configs.m_Smoothing.m_SmoothModel

            self.saturation_detection = configs.m_SaturationDetection

            self.trigger_mode = configs.m_Trigger.m_Mode
            self.trigger_source = configs.m_Trigger.m_Source 
            self.trigger_source_type = configs.m_Trigger.m_SourceType

            self.store_to_ram = configs.m_Control.m_StoreToRam


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


          
            

def read_metadata(file_path: str) -> Tuple[MetaData,
                                           AcquisitionParameters,
                                           SpectrometerParameters,
                                           DMDParameters]:
    """Reads metadata of a previous acquisition from JSON file.

    Args:
        file_path (str):
            Name of JSON file containing all metadata.

    Returns:
        Tuple[MetaData, AcquisitionParameters, SpectrometerParameters, 
        DMDParameters]:
            saved_metadata (MetaData):
                Metadata object read from JSON.
            saved_acquisition_params(AcquisitionParameters):
                AcquisitionParameters object read from JSON.
            saved_spectrometer_params(SpectrometerParameters):
                SpectrometerParameters object read from JSON.
            saved_dmd_params(DMDParameters):
                DMDParameters object read from JSON.
    """
    
    file = open(file_path,'r')
    data = json.load(file)
    file.close()
        
    for object in data:
        if object['class_description'] == 'Metadata':
            saved_metadata = MetaData.from_dict(object)
        if object['class_description'] == 'Acquisition parameters':
            saved_acquisition_params = AcquisitionParameters.from_dict(object)
            saved_acquisition_params.undo_readable_pattern_order()
        if object['class_description'] == 'Spectrometer parameters':
            saved_spectrometer_params = SpectrometerParameters.from_dict(object)
        if object['class_description'] == 'DMD parameters':
            saved_dmd_params = DMDParameters.from_dict(object)

    return (saved_metadata, saved_acquisition_params, 
            saved_spectrometer_params, saved_dmd_params)

def read_metadata_2arms(file_path: str) -> Tuple[MetaData,
                                           AcquisitionParameters,
                                           SpectrometerParameters,
                                           DMDParameters,
                                           CAM]:
    """Reads metadata of a previous acquisition from JSON file.

    Args:
        file_path (str):
            Name of JSON file containing all metadata.

    Returns:
        Tuple[MetaData, AcquisitionParameters, SpectrometerParameters, 
        DMDParameters]:
            saved_metadata (MetaData):
                Metadata object read from JSON.
            saved_acquisition_params(AcquisitionParameters):
                AcquisitionParameters object read from JSON.
            saved_spectrometer_params(SpectrometerParameters):
                SpectrometerParameters object read from JSON.
            saved_dmd_params(DMDParameters):
                DMDParameters object read from JSON.
    """
    
    file = open(file_path,'r')
    data = json.load(file)
    file.close()
        
    for object in data:
        if object['class_description'] == 'Metadata':
            saved_metadata = MetaData.from_dict(object)
        if object['class_description'] == 'Acquisition parameters':
            saved_acquisition_params = AcquisitionParameters.from_dict(object)
            saved_acquisition_params.undo_readable_pattern_order()
        if object['class_description'] == 'Spectrometer parameters':
            saved_spectrometer_params = SpectrometerParameters.from_dict(object)
        if object['class_description'] == 'DMD parameters':
            saved_dmd_params = DMDParameters.from_dict(object)
        if object['class_description'] == 'IDS camera configuration':
            saved_cam_params = CAM.from_dict(object)  
            saved_cam_params.undo_readable_class_CAM()
            

    return (saved_metadata, saved_acquisition_params, 
            saved_spectrometer_params, saved_dmd_params, saved_cam_params)

def save_metadata(metadata: MetaData, 
                  DMD_params: DMDParameters, 
                  spectrometer_params: SpectrometerParameters, 
                  acquisition_parameters: AcquisitionParameters) -> None:
    """Saves metadata to JSON file.

    Args:
        metadata (MetaData):
            Metadata concerning the experiment, paths, file inputs and file
            outputs.
        DMD_params (DMDParameters):
            Class containing DMD configurations and status.
        spectrometer_params (SpectrometerParameters):
            Object containing spectrometer configurations.
        acquisition_parameters (AcquisitionParameters):
            Object containing acquisition specifications and timing results.
    """

    path = Path(metadata.output_directory)
    with open(
        path / f'{metadata.experiment_name}_metadata.json',
        'w', encoding='utf8') as output:

        output_params = [
            metadata.to_dict(),
            DMD_params.to_dict(), 
            spectrometer_params.to_dict(), 
            AcquisitionParameters.readable_pattern_order(
                acquisition_parameters.to_dict())]

        json.dump(output_params,output,ensure_ascii=False,indent=4)
        
def save_metadata_2arms(metadata: MetaData, 
                  DMD_params: DMDParameters, 
                  spectrometer_params: SpectrometerParameters, 
                  camPar : CAM,
                  acquisition_parameters: AcquisitionParameters) -> None:
    """Saves metadata to JSON file.

    Args:
        metadata (MetaData):
            Metadata concerning the experiment, paths, file inputs and file
            outputs.
        DMD_params (DMDParameters):
            Class containing DMD configurations and status.
        spectrometer_params (SpectrometerParameters):
            Object containing spectrometer configurations.
        acquisition_parameters (AcquisitionParameters):
            Object containing acquisition specifications and timing results.
    """

    path = Path(metadata.output_directory)
    with open(
        path / f'{metadata.experiment_name}_metadata.json',
        'w', encoding='utf8') as output:

        output_params = [
            metadata.to_dict(),
            DMD_params.to_dict(), 
            spectrometer_params.to_dict(),
            CAM.readable_class_CAM(camPar.to_dict()),
            AcquisitionParameters.readable_pattern_order(acquisition_parameters.to_dict())]

        json.dump(output_params, output, ensure_ascii=False, indent=4)#, default=convert)
    
    # with open(path / f'{metadata.experiment_name}_metadata_cam.pkl', 'wb') as f:
    #     pickle.dump(camPar.__dict__, f)    

@dataclass_json
@dataclass
class func_path:
    def __init__(self, data_folder_name, data_name):        
        if not os.path.exists('../data/' + data_folder_name):
            os.makedirs('../data/' + data_folder_name)
            aborted = False
        else:
            res = input('Acquisition already exists, overwrite it ?[y/n]')
            if res == 'n':
                aborted = True
            else:
                aborted = False

        self.aborted = aborted
        self.subfolder_path = '../data/' + data_folder_name + '/' + data_name    
        self.overview_path = self.subfolder_path + '/overview'
        if not os.path.exists(self.overview_path):
            os.makedirs(self.overview_path)

        self.data_name = data_name
        self.data_path = self.subfolder_path + '/' + data_name
        self.had_reco_path = self.data_path + '_had_reco.npz'         
        self.fig_had_reco_path = self.overview_path + '/' + data_name   
        self.pathIDSsnapshot = Path(self.data_path + '_IDScam_before_acq.npy')
        self.pathIDSsnapshot_overview = self.overview_path + '/' + data_name + '_IDScam_before_acq.png'
        self.nn_reco_path = self.data_path + '_nn_reco.npz'
        self.fig_nn_reco_path = self.overview_path + '/' + data_name 

  