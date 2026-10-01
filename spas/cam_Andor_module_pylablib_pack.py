# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:48:30 2025

@author: mahieu
"""

# to read all attribute vlaues : cam.get_all_attribute_values()
encoding = 'utf-8'

import pylablib as pll
from pylablib.devices import Andor
import time
import numpy as np
from matplotlib import pyplot as plt
from typing import Optional
from dataclasses import dataclass, InitVar
from dataclasses_json import dataclass_json
from scipy import signal
from collections import deque
import cv2

# cam.get_all_attribute_values()

def init_cam_spat(SN : str = ''):
    """
    Initialize the saptial camera

    Parameters
    ----------
    SN : str, optional
        enter the serial number of the spatial camera.

    Returns
    -------
    cam_spat : object
        the object that drives the spatial camera.

    """
    
    lib_path = "C:/" + "Program Files" + "/" + "Andor SOLIS/*"
    pll.par["devices/dlls/andor_sdk3"] = lib_path

    cam_number = Andor.get_cameras_number_SDK3()
    cam_temp = Andor.AndorSDK3Camera(0)
    if cam_temp.get_attribute_value("SerialNumber") == SN:# cam spatial
        cam_spat = cam_temp
        del cam_temp
        # print('')
    elif cam_number > 0:
        cam_temp = Andor.AndorSDK3Camera(1)
        if cam_temp.get_attribute_value("SerialNumber") == SN:# cam spectral
            cam_spat = cam_temp
            del cam_temp
            
    cam_spat.arm = 'spatial'
    print('camera ANDOR, SN = ' + SN + ' connected and dedicated to the SPATIAL  arm')
    
    return cam_spat

def init_cam_spec(SN : str = ''):
    """
    Initialize the spectral camera

    Parameters
    ----------
    SN : str, optional
        enter the serial number of the spectral camera.

    Returns
    -------
    cam_spec : object
        the object that drives the spatial camera.

    """
    
    lib_path = "C:/" + "Program Files" + "/" + "Andor SOLIS/*"
    pll.par["devices/dlls/andor_sdk3"] = lib_path

    cam_number = Andor.get_cameras_number_SDK3()
    cam_temp = Andor.AndorSDK3Camera(0)
    if cam_temp.get_attribute_value("SerialNumber") == SN:# cam spatial
        cam_spec = cam_temp
        del cam_temp
        # print('')
    elif cam_number > 0:
        cam_temp = Andor.AndorSDK3Camera(1)
        if cam_temp.get_attribute_value("SerialNumber") == SN:# cam spectral
            cam_spec = cam_temp
            del cam_temp
            
    cam_spec.arm = 'spectral'
    print('camera ANDOR, SN = ' + SN + ' connected and dedicated to the SPECTRAL arm')
    
    return cam_spec

def disconnect_cam(cam):
    """
    disconnect the camera
    
    Parameters:
    -----------
        cam (obj): 
            a object to drive the Andor camera

    Returns
    -------
    None.

    """
    
    # cam_name = cam.get_device_name().decode(encoding)
    cam.close()
    if cam.arm == 'spatial':
        print('spatial  camera disconnected')
    elif cam.arm == 'spectral':
        print('spectral camera disconnected')
    

@dataclass_json
@dataclass
class cam_Parameters:
    """
    cam.get_all_attribute_values()
    """
    arm: Optional[str] = None
    exposure_time_s: Optional[float] = None
    exposure_time_µs: Optional[int] = None
    frame_rate: Optional[float] = None
    gain: Optional[str] = None

    width: Optional[int] = None
    height: Optional[int] = None
    offsetX: Optional[int] = None
    offsetY: Optional[int] = None
    binningX: Optional[int] = None
    binningY: Optional[int] = None
    
    bit_depth: Optional[str] = None   
    BytesPerPixel: Optional[str] = None
    
    Baseline: Optional[int] = None
    ExternalTriggerDelay: Optional[float] = None
    FanSpeed: Optional[str] = None # Off, Low, Medium, High
    FastAOIFrameRateEnable: Optional[bool] = None
    MetadataEnable: Optional[bool] = None
    MetadataTimestamp: Optional[bool] = None
    Overlap: Optional[bool] = None
    PreAmpGainSelector: Optional[str] = None # "Low" or "High"
    SimplePreAmpGainControl: Optional[str] = None 
    SpuriousNoiseFilter: Optional[bool] = None
    StaticBlemishCorrection: Optional[bool] = None
    TriggerMode: Optional[str] = None 
    
    bit_depth: Optional[str] = None
    simplePreAmpGainControl: Optional[str] = None
    encodPix: Optional[int] = None
    snapshot: Optional[bool] = None
    
    cam: InitVar[Andor.AndorSDK3Camera] = None
    
    class_description: str = None
    
    def __post_init__(self, cam: Optional[Andor.AndorSDK3Camera] = None):
        if cam == None:
            pass
        else:
            self.arm = cam.arm
            self.exposure_time_s = cam.get_attribute_value("ExposureTime")
            self.exposure_time_µs = self.exposure_time_s * 1e6
            self.FrameRate = cam.get_attribute_value("FrameRate")
            self.gain = cam.get_attribute_value("PreAmpGain")

            self.width  = cam.get_attribute_value("AOIWidth")
            self.height = cam.get_attribute_value("AOIHeight")
            self.offsetX  = cam.get_attribute_value("AOILeft")
            self.offsetY = cam.get_attribute_value("AOITop")            
            self.binningX = cam.get_attribute_value("AOIHBin")
            self.binningY = cam.get_attribute_value("AOIVBin")
            
            self.bit_depth = cam.get_attribute_value("BitDepth")
            self.BytesPerPixel = cam.get_attribute_value("BytesPerPixel")
            
            self.Baseline = cam.get_attribute_value("Baseline")
            self.ExternalTriggerDelay = cam.get_attribute_value("ExternalTriggerDelay")
            self.FanSpeed = cam.get_attribute_value("FanSpeed")
            self.FastAOIFrameRateEnable = cam.get_attribute_value("FastAOIFrameRateEnable")
            self.MetadataEnable = cam.get_attribute_value("MetadataEnable")
            self.MetadataTimestamp = cam.get_attribute_value("MetadataTimestamp")
            self.Overlap = cam.get_attribute_value("Overlap")
            self.PreAmpGainSelector = cam.get_attribute_value("PreAmpGainSelector")
            self.SimplePreAmpGainControl = cam.get_attribute_value("SimplePreAmpGainControl")
            self.SpuriousNoiseFilter = cam.get_attribute_value("SpuriousNoiseFilter")
            self.StaticBlemishCorrection = cam.get_attribute_value("StaticBlemishCorrection")
            self.TriggerMode = cam.get_attribute_value("TriggerMode")
            self.bit_depth = cam.get_attribute_value("BitDepth")
            self.simplePreAmpGainControl = cam.get_attribute_value("SimplePreAmpGainControl")
            
            encodPixel = cam.get_attribute_value("PixelEncoding")
            if encodPixel.find("Mono12Packed") == 0:
                encodPixel = 12
            elif encodPixel.find("Mono16") == 0:
                encodPixel = 16
            else:
                print("warning, problem to detect the enconding Pixel, default is 12 bit")
                encodPixel = 12
                
            self.encodPix = encodPixel
            self.snapshot = cam.snapshot            
            self.class_description = cam.arm + ' camera parameters'   
        

def setup_cam(cam: Andor.AndorSDK3Camera, expos_time: float = 0.1, ExternalTriggerDelay: float = 0, gain: int = 1, 
              baseline: int = 100, width: int = 2048, height: int = 2048, offsetX: int = 1, offsetY: int = 1, 
              binningX: int = 1, binningY: int = 1, encodPix: int = 12, snapshot: bool = False):
    """
    setup the Ximea camera

    Parameters
    ----------
    cam (obj): 
        a object to drive the Andor camera
    expos_time : float, optional
        the exposure time in s. The default is 0.1.
    ExternalTriggerDelay. float.
    the delay between the trigger received and to start the acquisition
    gain : int, optional
        the gain. The default is 1.
    baseline: int, optional
        the black level. default is 100
    width : int, optional
        the width of the ROI. The default is 2048.
    height : int, optional
        the height of the ROI. The default is 2048.
    offsetX : int, optional
        the offset of the ROI in the width direction. The default is 1.
    offsetY : int, optional
        the offset of the ROI in the height direction. The default is 1.
    binningX : int, optional
        the binning in the width direction. The default is 1.
    binningY : int, optional
        the binning in the height direction. The default is 1.
    encodPix : int, optional
        the encoding pixel. The default is 12 bits.
    snapshot: bool
        if false => acquire video, if True => acquire an image. default is False.

    Returns
    -------
    None.

    """
    ########################### setting binning ###############################
    binX = cam.get_attribute_value("AOIHBin")
    if binX != binningX:
        cam.set_attribute_value("AOIHBin", binningX)
        print('binning X set to: ' + str(binningX))
    
    binY = cam.get_attribute_value("AOIVBin")
    if binY != binningY:    
        cam.set_attribute_value("AOIVBin", binningY)
        print('binning Y set to: ' + str(binningY))
    ########################### setting ROI ###################################
    # width = cam.get_width()
    width_max = int(2048/binningX)
    # height = cam.get_height()
    height_max = int(2048/binningY)


    offsetX_cur = cam.get_attribute_value("AOILeft")
    offsetY_cur = cam.get_attribute_value("AOITop")

    if width > width_max:
        width_acc = width_max
    else:
        width_acc = width
    if height > height_max:
        height_acc = height_max
    else:
        height_acc = height
    offsetX_acc = offsetX
    offsetY_acc = offsetY
    
    # print('width acc    : ' + str(width_acc))
    # print('height acc   : ' + str(height_acc))
    # print('offset X acc : ' + str(offsetX_acc))
    # print('offset Y acc : ' + str(offsetY_acc))

    if offsetX_cur > 0 and offsetX_acc == 0:
        cam.set_attribute_value("AOILeft", offsetX_acc)  
        cam.set_attribute_value("AOIWidth", width_acc)  
    else:
        cam.set_attribute_value("AOIWidth", width_acc) 
        cam.set_attribute_value("AOILeft", offsetX_acc)  
         
    if offsetY_cur > 0 and offsetY_acc == 0: 
        cam.set_attribute_value("AOITop", offsetY_acc) 
        cam.set_attribute_value("AOIHeight", height_acc) 
    else:
        cam.set_attribute_value("AOIHeight", height_acc)  
        cam.set_attribute_value("AOITop", offsetY_acc)
         
    width_get = cam.get_attribute_value("AOIWidth")
    height_get = cam.get_attribute_value("AOIHeight")
    offsetX_get = cam.get_attribute_value("AOILeft")
    offsetY_get = cam.get_attribute_value("AOITop")
    
    print('width set to    : ' + str(width_get))
    print('height set to   : ' + str(height_get))
    print('offset X set to : ' + str(offsetX_get))
    print('offset Y set to : ' + str(offsetY_get))  
    ############### set the Spurious Noise Filter ############################# to reduce the "salt & pepper noise rendering
    SpuriousNoiseFilter = cam.get_attribute_value("SpuriousNoiseFilter")
    print("Spurious Noise Filter is", SpuriousNoiseFilter)
    if SpuriousNoiseFilter == False:
        cam.set_attribute_value("SpuriousNoiseFilter", True)
        SpuriousNoiseFilter = cam.get_attribute_value("SpuriousNoiseFilter")
        print("SpuriousNoiseFilter set to", SpuriousNoiseFilter)
    ############### set the Blemish Correction Filter ######################## to take off hot spike/unresponsive pixels
    StaticBlemishCorrection = cam.get_attribute_value("StaticBlemishCorrection")
    print("Static Blemish Correction is", StaticBlemishCorrection)
    if StaticBlemishCorrection == False:
        cam.set_attribute_value("StaticBlemishCorrection", True)
        StaticBlemishCorrection = cam.get_attribute_value("StaticBlemishCorrection")
        print("StaticBlemishCorrection set to", StaticBlemishCorrection)
    ################## read Pixel Readout Rate #################################
    cam.set_attribute_value("PixelReadoutRate",'100 MHz')# other possibiliti is "270 MHz", it is faster and noiser
    PixelReadoutRate = cam.get_attribute_value("PixelReadoutRate")
    print("Pixel Readout Rate =", PixelReadoutRate)
    #################### read Trigger Mode ####################################
    # cam.set_attribute_value("TriggerMode", "Internal")    
    # TriggerMode = cam.get_attribute_value("TriggerMode")
    # print("Trigger Mode is :", TriggerMode)
    TriggerMode = cam.get_attribute_value("TriggerMode")
    if TriggerMode != "External":
        cam.set_attribute_value("TriggerMode", "External")
        TriggerMode = cam.get_attribute_value("TriggerMode")
        print("Trigger Mode is :", TriggerMode)
    # if TriggerMode != "Internal":
    #     cam.set_attribute_value("TriggerMode", "Internal")
    #     TriggerMode = cam.get_attribute_value("TriggerMode")
    #     print("Trigger Mode is :", TriggerMode)
    ################# set External Trigger Delay ##############################
    ExternalTriggerDelay_current = cam.get_attribute_value("ExternalTriggerDelay")
    if ExternalTriggerDelay_current != ExternalTriggerDelay:
        cam.set_attribute_value("ExternalTriggerDelay", )
        print("External Trigger Delay =", ExternalTriggerDelay, "!!!  need to check the unity" )
    else:
        print("External Trigger Delay already set to =", ExternalTriggerDelay_current)
    ############### read the SimplePreAmpGainControl ##########################
    if encodPix == 12:
        SimplePreAmpGainControl = "12-bit (low noise)"
    elif encodPix == 16:
        SimplePreAmpGainControl = "16-bit (low noise & high well capacity)"
    else:
        print("Warning, encoding pixel has a bad entry value. default is taken (12 bits)")
        encodingPixel = "Mono12Packed"
    
    cam.set_attribute_value("SimplePreAmpGainControl", SimplePreAmpGainControl)    
    SimplePreAmpGainControl = cam.get_attribute_value("SimplePreAmpGainControl")
    print("Simple PreAmp Gain Control =", SimplePreAmpGainControl)
    ################# read the Bit depth ######################################   
    BitDepth = cam.get_attribute_value("BitDepth")
    print("BitDepth =", BitDepth)
    ##################### Encoding Pixel ######################################
    if encodPix == 12:
        encodingPixel = "Mono12Packed"
    elif encodPix == 16:
        encodingPixel = "Mono16"
    else:
        print("Warning, encoding pixel has a bad entry value. default is taken (12 bits)")
        encodingPixel = "Mono12Packed"
        
    cam.set_attribute_value("PixelEncoding", encodingPixel) 
    pixelEncoding = cam.get_attribute_value("PixelEncoding")
    print("Pixel Encoding =", pixelEncoding)
    #################### setting the exposure time ###########################
    # NB: the exposure time must be an interger in s 
    if PixelReadoutRate == "100 MHz":
        exposure_mini = 7.2e-5
        # exposure_mini = 0.000984
    elif PixelReadoutRate == "270 MHz":
        exposure_mini = 2.89e-5
    else:
        print('Warning, the PixelReadoutRate is different tahn 100 or 270 MHZ where min Expose Time is not define, plaese check !!!!')
    exposure_maxi = 4.9
    exposure_time = expos_time #round(expos_time * 1000)
    if exposure_time < exposure_mini:
        exposure_time = exposure_mini
        print('the exposure time is below the minimum value, it is set to ' + str(exposure_time))
    if exposure_time > exposure_maxi:
        exposure_time = exposure_maxi
        print('the exposure time is above the maximum value, it is set to ' + str(exposure_time))
        
    cam.set_attribute_value("ExposureTime", exposure_time)
    exposure_time_get = cam.get_attribute_value("ExposureTime")
    print('exposure time set to : ' + str(exposure_time_get) + ' s')
    ######################### get the frame rate ##############################
    current_frame_rate = cam.get_attribute_value("FrameRate")
    print('frame rate = ' + str(current_frame_rate) + ' fps')
    # print('!!! warning, at this moment, the frame rate cannot be set')
    # cam.set_framerate(frame_rate)
    # current_frame_rate = cam.get_framerate()
    # print('new frame rate set to : ' + str(current_frame_rate))
    ########################## setting gain ###################################
    curent_gain = cam.get_attribute_value("PreAmpGain")
    if curent_gain != 'x' + str(gain):
        cam.set_attribute_value("PreAmpGain", 'x' + str(gain))
        get_gain = cam.get_attribute_value("PreAmpGain")
        print('the gain is set to : ' + str(get_gain))
    ################♠ acquisition mode: snapshot or video #####################
    cam.snapshot = snapshot
    # ######################## set buffer #######################################
    # if cam.snapshot == False:
    #     cam.set_attribute_value("BufferSize", 256 * 2) # 256 est un nombre arbitraire qui devrait être: acquisition_params.pattern_amount que l'on multiplie par 2 pour avoir suffisamment de mémoire
    # else:
    #     cam.set_attribute_value("BufferSize", 2)
    
    return cam_Parameters(cam = cam)
        

def snapshot_cam(cam, tilt_image: bool = False):
    """
    take a snapshot of the camera
    
    Parameters:
    -----------
        cam (obj): 
            a object to drive the Ximea camera
         tilt_image (bool):
             Tilte the image as display on the DMD. Default is False.
    Returns
    -------
    None.

    """
    def rebin(arr, new_shape):
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])
        return arr.reshape(shape).mean(-1).mean(1)
    
    exp_time = cam.get_attribute_value("ExposureTime")        
    image_bit_depth_str = cam.get_attribute_value("BitDepth")
    image_bit_depth = int(image_bit_depth_str[:image_bit_depth_str.index(' Bit')])

    data = cam.snap(timeout = exp_time + 5)
    ########################• 2d median filter ################################
    data = signal.medfilt2d(data, kernel_size=3)
    ######################## check saturation #################################
    data_max = np.max(data)
    if (data_max >= 255 and image_bit_depth == 8) or (data_max >= 1023 and image_bit_depth == 10) or (data_max >= 4095 and image_bit_depth == 12) or (data_max >= 65535 and image_bit_depth == 16):
        print('!!!!!!!!!! Warning, saturation detected !!!!!!!!!!!!')
    ########################## tilt image ####################################
    if tilt_image:
        # data = np.flip(np.flip(data, axis = 1), axis = 0)
        data = np.rot90(data, k=1, axes=(0,1))
        print('image tilted')
    ########################## print snapshot #################################
    plt.figure()
    plt.imshow(data)
    plt.colorbar()
    plt.title(cam.arm + ' cam')
    plt.xlabel('X (Width)')
    plt.ylabel('Y (Height)')
    
    Sig = np.max(data[round(data.shape[0]/2 - 100):round(data.shape[0]/2 + 100), round(data.shape[1]/2 - 100):round(data.shape[1]/2 + 100)])
    noise = np.std(data[round(50/(data.shape[0]/cam.get_attribute_value("AOIHeight"))):round(150/(data.shape[0]/cam.get_attribute_value("AOIHeight"))), round(data.shape[1]/2 - 100):round(data.shape[1]/2 + 100)].flatten())
    maxi = np.max(data)
    print('Original :')
    print('     max   = ' + str(maxi))
    print('     Sig   = ' + str(Sig))
    print('     Noise = ' + str(noise))
    print('     SNR   = ' + str(Sig/noise))

    return data    


def display_cam(cam, cam_params, display_max: bool = False, 
                display_integral: bool = False, display_profile: bool = False):
    """
    Continuous image display of a camera with optional integral/mean curve using OpenCV.

    Parameters:
    -----------
        cam (obj):
            Object to drive the Ximea camera
        cam_params (obj):
            Camera parameters (width, height, etc.)
        display_max (bool):
            Display the maximum value in the image. Default is False.
        display_integral (bool):
            Display the mean value of the image as a curve. Default is False.
        display_profile (bool):
            Display the profile image to see the edge of the light sheet. Default is False.
    Returns:
    -------
        None
    """

    # Define the output window size
    width = cam_params.width
    height = cam_params.height
    ratio = width / height
    height_win = 900
    width_win = int(height_win * ratio)

    image_bit_depth_str = cam.get_attribute_value("BitDepth")
    image_bit_depth = int(image_bit_depth_str[:image_bit_depth_str.index(' Bit')])

    try:
        # Create OpenCV window for the camera
        if cam.arm == 'spatial':
            window_name = "Camera of the Spatial Arm"
        elif cam.arm == 'spectral':
            window_name = "Camera of the Spectral Arm"

        cv2.namedWindow(window_name)

        # Trackbar callback (does nothing)
        def nothing(x):
            pass

        # Exposure time setup
        min_exposure_time = 0.000984 * 1e6
        max_exposure_time = 4.9 * 1e6
        current_exposure_time = cam.get_attribute_value("ExposureTime")
        t_wait = current_exposure_time

        # Integral curve setup (if enabled)
        if display_integral:
            max_points = 800
            vector = deque(maxlen=max_points)
            curve_height = 800
            curve_img = np.zeros((curve_height, max_points), dtype=np.uint8)
            cv2.namedWindow("Integral Curve")
            cv2.resizeWindow("Integral Curve", max_points, curve_height)

        first_passage = True
        first_passage2 = True        
        maxii = 0
        
        if display_profile:
            # curve_height = 800
            # first_pass = False # for the profile dispay
            curve_height = 200  # Hauteur de l'image de la courbe
            fixed_width = 800    # Largeur fixe de la fenêtre
            prof_width_win = 500     # Largeur de la fenêtre principale (à adapter)
            # Initialiser l'image pour la courbe (noire, 1 canal)
            curve_img = np.zeros((curve_height, fixed_width), dtype=np.uint8)
            
        def smooth_profile(profile, window_size=10):
            """Applique un lissage par moyenne mobile."""
            smoothed = np.convolve(profile, np.ones(window_size)/window_size, mode='same')
            return smoothed
            
        # Start data acquisition
        print('Start acquisition...\n')
        cam.start_acquisition()

        while True:
            # Acquire image
            data = cam.snap(timeout=current_exposure_time + 5)

            if cam.arm == 'spatial':
                data = np.rot90(data, k=1, axes=(0, 1))

            data_8b = cv2.convertScaleAbs(data, alpha=(255.0 / (2**image_bit_depth - 1)))

            # Calculate max value
            maxi = np.max(data_8b)
            if maxi != maxii:
                if display_max:
                    print("max = " + str(maxi))
                maxii = maxi

            # Saturation detection
            if maxi == 255 and first_passage2:
                print('Saturation detected')
                first_passage2 = False
            elif maxi < 255 and not first_passage2:
                print('No more saturation')
                first_passage2 = True

            # Initialize trackbars on first pass
            if first_passage:
                cv2.createTrackbar('Brightness', window_name, maxi, 510, nothing)
                cv2.createTrackbar('Exp time (µs)', window_name, int(current_exposure_time * 1e6), 50000, nothing)
                first_passage = False

            # Get trackbar values
            brightness = cv2.getTrackbarPos('Brightness', window_name)
            exposure_time = cv2.getTrackbarPos('Exp time (µs)', window_name)

            # Clamp exposure time
            if exposure_time < min_exposure_time:
                exposure_time = min_exposure_time
            elif exposure_time > max_exposure_time:
                print('Maximum exposure time set to 4.9 s')
                exposure_time = max_exposure_time

            cam.set_attribute_value("ExposureTime", exposure_time / 1e6)

            # Process image
            data_64b = data_8b.astype(np.float64)
            data2 = data_64b * brightness / maxi
            data_8b = data2.astype(np.uint8)
            data_8b_resize = cv2.resize(data_8b, (width_win, height_win))

            # Display camera image
            cv2.imshow(window_name, data_8b_resize)
            cv2.moveWindow(window_name, 0, 0)

            # Display integral curve (if enabled)
            if display_integral:
                integral = np.mean(data_64b)
                vector.append(integral)

                # Normalize values for display
                y_values = np.array(vector)
                y_min, y_max = min(y_values), max(y_values)
                y_range = y_max - y_min if y_max != y_min else 1

                # Clear curve image
                curve_img.fill(0)

                # Draw curve
                for i in range(1, len(vector)):
                    x1 = i - 1
                    x2 = i
                    y1 = int(curve_height - (vector[x1] - y_min) / y_range * curve_height)
                    y2 = int(curve_height - (vector[x2] - y_min) / y_range * curve_height)
                    cv2.line(curve_img, (x1, y1), (x2, y2), 255, 2)

                # Draw axes
                cv2.line(curve_img, (0, curve_height - 1), (max_points - 1, curve_height - 1), 255, 1)
                cv2.line(curve_img, (0, 0), (0, curve_height - 1), 255, 1)

                # Display curve
                cv2.imshow("Integral Curve", curve_img)
                cv2.moveWindow("Integral Curve", width_win + 20, 0)
                
            # curve_img = np.zeros((curve_height, len(data_64b)), dtype=np.uint8)
            
            if display_profile:
                # define offset and thickness of the profile
                offset = int(data_64b.shape[0]/2)
                half_thick = 50
                # Extraire le profil (colonne centrale)
                data_rogne = data[offset - half_thick:offset + half_thick, :]
                
                profile = np.mean(data_rogne, axis=0)
            
                # Lisser le profil
                smoothed_profile = profile#smooth_profile(profile, window_size=10)
                smoothed_profile = smoothed_profile[10: len(profile) - 10]
                # Normaliser le profil lissé pour l'affichage
                y_min, y_max = np.min(smoothed_profile), np.max(smoothed_profile)
                y_range = y_max - y_min if y_max != y_min else 1
            
                # Effacer l'image de la courbe
                curve_img.fill(0)
            
                # Échantillonner le profil pour qu'il tienne dans fixed_width
                step = len(smoothed_profile) / fixed_width
                points = []
                for i in range(fixed_width):
                    idx = int(i * step)
                    if idx >= len(smoothed_profile):
                        break
                    x = i
                    y = int(curve_height - (smoothed_profile[idx] - y_min) / y_range * curve_height)
                    points.append((x, y))
            
                # Tracer le profil lissé
                if len(points) > 1:
                    cv2.polylines(curve_img, [np.array(points, dtype=np.int32)], False, 255, 1)
            
                # Tracer les axes (optionnel)
                cv2.line(curve_img, (0, curve_height - 1), (fixed_width - 1, curve_height - 1), 255, 1)  # Axe horizontal
                cv2.line(curve_img, (0, 0), (0, curve_height - 1), 255, 1)  # Axe vertical
            
                # Afficher le profil
                cv2.imshow("Profile", curve_img)
                cv2.moveWindow("Profile", width_win + 20, 0)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        cam.stop_acquisition()
        print('Stopping acquisition...')

# # def display_cam(cam, display_max: bool = False, binningX: int = 1, binningY: int = 1):
# def display_cam(cam, cam_params, display_max: bool = False, display_integral: bool = False):
#     """
#     Continuous image display of a camera
    
#     Parameters:
#     -----------
#         cam (obj): 
#             a object to drive the Ximea camera  
#         display_max (bool):
#             display the maximum value in the image. Default is True.
#         display_integral (bool):
#             display the mean value of the image. Default is False.
#     Returns:
#     -------
#         None
#     """
    
#     # Define the output window size
#     width = cam_params.width
#     height = cam_params.height
#     ratio = width / height
#     height_win = 900
#     width_win = int(height_win * ratio)
    
#     image_bit_depth_str = cam.get_attribute_value("BitDepth")
#     image_bit_depth = int(image_bit_depth_str[:image_bit_depth_str.index(' Bit')])
#     # image_bit_depth = 12
    
#     try:
#         import cv2
#         # Creating a cv2 window
#         if cam.arm == 'spatial':
#             window_name = "Camera of the Spatial Arm"
#         elif cam.arm == 'spectral':
#             window_name = "Camera of the Spectral Arm"
            
#         cv2.namedWindow(window_name) 
        
#         # Create a function 'nothing' for creating trackbar 
#         def nothing(x): 
#             pass
        
#         min_exposure_time = 0.000984 * 1e6
#         max_exposure_time = 4.9 * 1e6
#         current_exposure_time = cam.get_attribute_value("ExposureTime")
#         t_wait = current_exposure_time
#         print('wait time = ' + str(t_wait))
        
#         if display_integral == True:
#             # Fenêtre de 50 valeurs
#             max_points = 50
#             vector = deque(maxlen=max_points)
            
#             plt.ion()
#             fig, ax = plt.subplots()
#             plt.show(block=False)
#             line, = ax.plot([], [], 'o-')        
#             # limites fixes pour la fenêtre glissante
#             ax.set_xlim(0, max_points)
#             ax.set_ylim(0, 255)
    
#         first_passage = True
        
#         #start data acquisition
#         print('Start acquisition...\n')
#         cam.start_acquisition()
        
#         first_passage2 = True
#         # data_center_old = 0
#         maxii = 0
        
#         while True:
            
#             data = cam.snap(timeout = current_exposure_time + 5)
            
#             if cam.arm == 'spatial':
#                 data = np.rot90(data, k=1, axes=(0,1))
                
#             data_8b = cv2.convertScaleAbs(data, alpha=(255.0/(2**image_bit_depth - 1)))    
            
            
#             maxi = np.max(data_8b)
#             print("max = " + str(maxi))
#             if maxi != maxii:
#                 if display_max:
#                     print("max = " + str(maxi))
#                 maxii = maxi
            
#             if maxi == 255 and first_passage2 == True:
#                 print('saturation detected')
#                 first_passage2 = False
#             elif maxi < 255 and first_passage2 == False:
#                 print('No more saturation')
#                 first_passage2 = True
            
#             if first_passage == True:
#                 maxi = np.max(data_8b)
#                 print('maxi = ' + str(maxi))
#                 print('press "q" to exit')
#                 # Creating trackbars for color change 
#                 cv2.createTrackbar('Brightness', window_name, maxi, 510, nothing) 
    
#                 cv2.createTrackbar('Exp time (µs)', window_name, int(current_exposure_time*1e6), 50000, nothing) 
    
#                 first_passage = False

            
#             # Get current positions of trackbar 
#             brightness = cv2.getTrackbarPos('Brightness', window_name) 
            
#             # Tune exposure time
#             exposure_time = cv2.getTrackbarPos('Exp time (µs)', window_name)             
#             if exposure_time < min_exposure_time:
#                 exposure_time = min_exposure_time                
#             elif exposure_time > max_exposure_time:
#                 print('maximum exposure time set to 4.9 s')
#                 exposure_time = max_exposure_time
            
#             cam.set_attribute_value("ExposureTime", exposure_time/1e6)
            
#             data_64b = data_8b.astype(np.float64)           
#             data2 = data_64b*brightness/maxi
#             data_8b = data2.astype(np.uint8)
#             data_8b_resize = cv2.resize(data_8b, (width_win, height_win)) 

#             cv2.imshow(window_name, data_8b_resize)
#             cv2.moveWindow(window_name, 0, 0)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 cv2.destroyWindow(window_name)
#                 #stop data acquisition
#                 print('cam: Stopping acquisition...')
#                 cam.stop_acquisition()

#                 break
            
#             if display_integral == True:
#                 integral = np.mean(np.mean(data_64b, axis=1), axis=0)
#                 vector.append(integral)
                
#                 x = list(range(len(vector)))
#                 y = list(vector)
            
#                 line.set_data(x, y)
                
#                 ax.relim()
#                 ax.autoscale_view()
    
#                 # on ajuste seulement Y
#                 ax.set_ylim(min(y)-0.5, max(y)+0.5)
            
#                 fig.canvas.draw()
#                 fig.canvas.flush_events()
#                 plt.pause(0.01)                
                
#                 manager = plt.get_current_fig_manager()
#                 manager.window.wm_geometry("+950+0")
                
#     except:
#         cv2.destroyWindow(window_name)
#         cam.stop_acquisition()
#         print('try function encoutered a exception')

    
        
        
        
        
        
        
        
        
        
        
        
        