# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:48:30 2025

@author: mahieu
"""
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
    
    image_data_bit_depth: Optional[str] = None
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

            self.snapshot = cam.snapshot            
            self.class_description = cam.arm + ' camera parameters'   
        

def setup_cam(cam: Andor.AndorSDK3Camera, expos_time: float = 0.1, ExternalTriggerDelay: float = 0, gain: int = 1, baseline: int = 100, 
              width: int = 2048, height: int = 2048, offsetX: int = 1, offsetY: int = 1, binningX: int = 1, binningY: int = 1, snapshot: bool = False):
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
    PixelReadoutRate = cam.get_attribute_value("PixelReadoutRate")
    print("Pixel Readout Rate =", PixelReadoutRate)
    print("for information, 216 or 540 MHz is not available")
    #################### read Trigger Mode ####################################
    # cam.set_attribute_value("TriggerMode", "Internal")    
    # TriggerMode = cam.get_attribute_value("TriggerMode")
    # print("Trigger Mode is :", TriggerMode)
    TriggerMode = cam.get_attribute_value("TriggerMode")
    if TriggerMode != "External":
        cam.set_attribute_value("TriggerMode", "External")
        TriggerMode = cam.get_attribute_value("TriggerMode")
        print("Trigger Mode is :", TriggerMode)
    ################# set External Trigger Delay ##############################
    ExternalTriggerDelay_current = cam.get_attribute_value("ExternalTriggerDelay")
    if ExternalTriggerDelay_current != ExternalTriggerDelay:
        cam.set_attribute_value("ExternalTriggerDelay", )
        print("External Trigger Delay =", ExternalTriggerDelay, "!!!  need to check the unity" )
    else:
        print("External Trigger Delay already set to =", ExternalTriggerDelay_current)
    ############### read the SimplePreAmpGainControl ##########################
    SimplePreAmpGainControl = cam.get_attribute_value("SimplePreAmpGainControl")
    print("Simple PreAmp Gain Control =", SimplePreAmpGainControl)
    ################# read the Bit depth ######################################
    BitDepth = cam.get_attribute_value("BitDepth")
    print("BitDepth =", BitDepth)
    
    #################### setting the exposure timre ###########################
    # # NB: the exposure time must be an interger in s 
    # exposure_time = round(expos_time * 1000)
    # if exposure_time < cam.get_exposure_minimum():
    #     exposure_time = cam.get_exposure_minimum()
    #     print('the exposure time is below the minimum value, it is set to ' + str(cam.get_exposure_minimum()))
    # if exposure_time > cam.get_exposure_maximum():
    #     exposure_time = cam.get_exposure_maximum()
    #     print('the exposure time is above the maximum value, it is set to ' + str(cam.get_exposure_maximum()))
        
    cam.set_attribute_value("ExposureTime", expos_time)
    exposure_time_get = cam.get_attribute_value("ExposureTime")
    print('exposure time set to : ' + str(exposure_time_get) + ' s')
    ######################### get the frame rate ##############################
    current_frame_rate = cam.get_attribute_value("FrameRate")
    print('frame rate = ' + str(current_frame_rate) + ' fps')
    # print('!!! warning, at this moment, the frame rate cannot be set')
    # cam.set_framerate(frame_rate)
    # current_frame_rate = cam.get_framerate()
    # print('new frame rate set to : ' + str(current_frame_rate))
    # ########################### get bandwidth #################################
    # cam.set_limit_bandwidth_mode('XI_ON')
    # interface_data_rate = cam.get_limit_bandwidth_maximum()
    # camera_data_rate = int(interface_data_rate / cameras_nbr)

    # min_data_rate_cam = cam.get_limit_bandwidth_minimum()
    # max_data_rate_cam = cam.get_limit_bandwidth_maximum()
    
    # if camera_data_rate < min_data_rate_cam:
    #     camera_data_rate = min_data_rate_cam
    #     print('camera_data_rate is below the minimum value, it is set to its miminum value')        
    # elif camera_data_rate > max_data_rate_cam:
    #     camera_data_rate = max_data_rate_cam
    #     print('camera_data_rate is above the maximum value, it is set to its maxinum value')
   
    # cam.set_limit_bandwidth(camera_data_rate)
    # print('BandWidth = ' + str(cam.get_limit_bandwidth()))
    ########################## set buffer #####################################
    # cam.set_buffer_policy('XI_BP_SAFE')#
    # cam.set_acq_buffer_size(int(cam.get_acq_buffer_size_maximum()/4)) # divide by 4 because if higher, we lose triggers, to set max, you need too wait 2.6s between star_acquisition and receive the first trig (DMD.run), to set max/2 => wait 1.5s, max/4 => wait 1s

    # print('buffer size = ' + str(cam.get_acq_buffer_size()))
    # print('min buffers queue size = ' + str(cam.get_buffers_queue_size_minimum()))
    # print('max buffers queue size = ' + str(cam.get_buffers_queue_size_maximum()))
    
    # cam.set_buffers_queue_size(cam.get_buffers_queue_size_maximum())
    
    # buffers_queue_size = cam.get_buffers_queue_size()
    # print('buffers queue size  = ' + str(buffers_queue_size))
    
    ########################## setting gain ###################################
    curent_gain = cam.get_attribute_value("PreAmpGain")
    if curent_gain != 'x' + str(gain):
        cam.set_attribute_value("PreAmpGain", 'x' + str(gain))
        get_gain = cam.get_attribute_value("PreAmpGain")
        print('the gain is set to : ' + str(get_gain))
    ######################## setting white balance ############################
    
    ################# acquire waiting an internal trigger #####################
    # gpi_mode = 'XI_GPI_TRIGGER'
    # cam.set_gpi_mode(gpi_mode)
    # trigger_source = 'XI_TRG_EDGE_RISING'
    # cam.set_trigger_source(trigger_source)
    # trigger_selector = 'XI_TRG_SEL_FRAME_START'
    # cam.set_trigger_selector(trigger_selector)
    # cam.set_trigger_overlap('XI_TRG_OVERLAP_OFF') # in the case of 'XI_TRG_OVERLAP_PREV_FRAME', a stray line appear in the image => desastrous for the Hadamard reco
    ################♠ acquisition mode: snapshot or video #####################
    cam.snapshot = snapshot
    
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
    if (data_max >= 255 and image_bit_depth == 8) or (data_max >= 1023 and image_bit_depth == 10) or (data_max >= 65535 and image_bit_depth == 16):
        print('!!!!!!!!!! Warning, saturation detected !!!!!!!!!!!!')
    ########################## tilt image ####################################
    if tilt_image:
        data = np.flip(np.flip(data, axis = 1), axis = 0)
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


# def display_cam(cam, display_max: bool = False, binningX: int = 1, binningY: int = 1):
def display_cam(cam, cam_params, display_max: bool = False):
    """
    Continuous image display of a camera
    
    Parameters:
    -----------
        cam (obj): 
            a object to drive the Ximea camera  
        display_max (bool):
            display the maximum value in the image. Default is True.
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
    
    try:
        # cam.set_buffers_queue_size(2)
        import cv2
        # Creating a cv2 window
        if cam.arm == 'spatial':
            window_name = "Camera of the Spatial Arm"
        elif cam.arm == 'spectral':
            window_name = "Camera of the Spectral Arm"
            
        cv2.namedWindow(window_name) 
        
        # Create a function 'nothing' for creating trackbar 
        def nothing(x): 
            pass
        
        min_exposure_time = 0
        max_exposure_time = 4.9 * 1e6
        current_exposure_time = cam.get_attribute_value("ExposureTime")
        t_wait = current_exposure_time
        print('wait time = ' + str(t_wait))
        
        # gain = cam.get_gain(PreAmpGain)
        # gain_min = 1
        # gain_max = cam.get_gain_maximum()
    
        first_passage = True
        
        #start data acquisition
        print('Start acquisition...\n')
        cam.start_acquisition()
        
        first_passage2 = True
        data_center_old = 0
        maxii = 0
        while True:
            # time.sleep(t_wait) # Sleep for 1 seconds
            # time.sleep(1)
            # cam.wait_for_frame()
            # data = cam.read_newest_image()
            
            data = cam.snap(timeout = current_exposure_time + 5)
            data_8b = cv2.convertScaleAbs(data, alpha=(255.0/4095))           
            
            maxi = np.max(data_8b)
            print("max = " + str(maxi))
            if maxi != maxii:
                if display_max:
                    print("max = " + str(maxi))
                maxii = maxi
            
            if maxi == 255 and first_passage2 == True:
                print('saturation detected')
                first_passage2 = False
            elif maxi < 255 and first_passage2 == False:
                print('No more saturation')
                first_passage2 = True
            
            if first_passage == True:
                maxi = np.max(data_8b)
                print('maxi = ' + str(maxi))
                print('press "q" to exit')
                # Creating trackbars for color change 
                cv2.createTrackbar('Brightness', window_name, maxi, 510, nothing) 
    
                cv2.createTrackbar('Exp time (µs)', window_name, int(current_exposure_time*1e6), 50000, nothing) 
    
                first_passage = False

            
            # Get current positions of trackbar 
            brightness = cv2.getTrackbarPos('Brightness', window_name) 
            
            # Tune exposure time
            exposure_time = cv2.getTrackbarPos('Exp time (µs)', window_name)             
            if exposure_time < min_exposure_time:
                exposure_time = min_exposure_time                
            elif exposure_time > max_exposure_time:
                print('maximum exposure time set to 4.9 s')
                exposure_time = max_exposure_time
            
            cam.set_attribute_value("ExposureTime", exposure_time/1e6)
            

            data_64b = data_8b.astype(np.float64)
            data2 = data_64b*brightness/maxi
            data_8b = data2.astype(np.uint8)
            data_8b_resize = cv2.resize(data_8b, (width_win, height_win)) 

            cv2.imshow(window_name, data_8b_resize)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow(window_name)
                #stop data acquisition
                print('cam: Stopping acquisition...')
                cam.stop_acquisition()

                break
        
    except:
        cv2.destroyWindow(window_name)
        cam.stop_acquisition()
        print('try function encoutered a exception')


# def counter_trigger(cam):
#     """
#     Arg:
#         cam (obj): 
#             a object to drive the Ximea camera
#     Returns:
#         a tuple containing counter of the trigger skipped and received
#     """
#     cam.set_counter_selector('XI_CNT_SEL_TRANSPORT_SKIPPED_FRAMES')
#     transport_skipped_trig = cam.get_counter_value()
#     cam.set_counter_selector('XI_CNT_SEL_API_SKIPPED_FRAMES')
#     api_skipped_trig = cam.get_counter_value()
#     cam.set_counter_selector('XI_CNT_SEL_TRANSPORT_TRANSFERRED_FRAMES')
#     transported_frames = cam.get_counter_value()
    
#     return [transport_skipped_trig, api_skipped_trig, transported_frames]        
        
        
        
        
        
        
        
        
        
        
        
        