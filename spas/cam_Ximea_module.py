# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:48:30 2025

@author: mahieu
"""
encoding = 'utf-8'

from ximea import xiapi
import time
import numpy as np
from matplotlib import pyplot as plt
from typing import Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json


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
    
    cam_spat = xiapi.Camera(dev_id = 0)
    cam_spat.open_device_by_SN(SN)
    cam_spat.arm = 'spatial'
    print('Color camera connected and dedicated to the spatial  arm')
    
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
    
    cam_spec = xiapi.Camera(dev_id = 1)
    cam_spec.open_device_by_SN(SN)
    cam_spec.arm = 'spectral'
    print('B & W camera connected and dedicated to the spectral arm')
    
    return cam_spec

def disconnect_cam(cam):
    """
    disconnect the camera
    
    Parameters:
    -----------
        cam (obj): 
            a object to drive the Ximea camera

    Returns
    -------
    None.

    """
    
    # cam_name = cam.get_device_name().decode(encoding)
    cam.close_device()
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
    exposure_time_µs: Optional[int] = None
    frame_rate: Optional[float] = None
    gain: Optional[float] = None
    is_auto_wb: Optional[bool] = None
    wb_red: Optional[float] = None
    wb_green: Optional[float] = None
    wb_blue: Optional[float] = None
    gammaY: Optional[float] = None
    gammaC: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    offsetX: Optional[int] = None
    offsetY: Optional[int] = None
    binningX: Optional[int] = None
    binningY: Optional[int] = None
    sensor_bit_depth: Optional[str] = None
    output_bit_depth: Optional[str] = None
    image_data_bit_depth: Optional[str] = None
    
    
    class_description: str = 'CAMERA parameters'
    
    def __init__(self, cam):
        if cam == None:
            print('cam is None in init of the class: cam_Parameters')
        else:
            self.arm = cam.arm
            self.exposure_time_µs = cam.get_exposure()
            self.gain = round(cam.get_framerate(), 2)
            self.gain = round(cam.get_gain(), 2)
            if cam.get_device_name().decode(encoding) == 'CB013CG-LX-X8G3':
                self.is_auto_wb = cam.is_auto_wb() 
                self.wb_red = round(cam.get_wb_kr(), 2)
                self.wb_green = round(cam.get_wb_kg(), 2)
                self.wb_blue= round(cam.get_wb_kb(), 2)
            self.gammaY = round(cam.get_gammaY(), 2)
            self.gammaC = round(cam.get_gammaC(), 2)
            self.width  = cam.get_width()
            self.height = cam.get_height()
            self.offsetX  = cam.get_offsetX()
            self.offsetY = cam.get_offsetY()
            self.binningX = cam.get_binning_horizontal()
            self.binningY = cam.get_binning_vertical()
            self.sensor_bit_depth = cam.get_sensor_bit_depth()
            self.output_bit_depth = cam.get_output_bit_depth()
            self.image_data_bit_depth = cam.get_image_data_bit_depth()
            
            
            
            
            
    # def __post_init__(self, cam: Optional = None):
    #     print('in post init')
    #     if cam == None:
    #         print('cam is None in post init')
    #     else:
    #         print('cam is not None in post init')
    #         self.exposure_time_µs = cam.get_exposure()   
        

def setup_cam(cam: object, cameras_nbr: int = 2, expos_time: float = 1, frame_rate: int = 3700, gain: float = 0, black_level: int = 4, 
              auto_wb: bool = True, gammaY: float = 0.3, width: int = 1280, height: int = 864, offsetX: int = 0, offsetY: int = 0, 
              binningX: int = 1, binningY: int = 1):
    """
    setup the Ximea camera

    Parameters
    ----------
    cam (obj): 
        a object to drive the Ximea camera
    cameras_nbr : int, optional
        the number of camera connected on the same controller. The default is 2.
    expos_time : float, optional
        the exposure time in ms. The default is 1.
    frame_rate : int, optional
        the acquisition frame rate
    gain : float, optional
        the gain. The default is 0.
    black_level: int, optional
        the black level. default is 4
    auto_wb : bool, optional
        auto white blance. The default is True.
    gammaY: float, optional
        the luminosity gamma. The default is 0.3.
    width : int, optional
        the width of the ROI. The default is 1280.
    height : int, optional
        the height of the ROI. The default is 864.
    offsetX : int, optional
        the offset of the ROI in the width direction. The default is 0.
    offsetY : int, optional
        the offset of the ROI in the height direction. The default is 0.
    binningX : int, optional
        the binning in the width direction. The default is 1.
    binningY : int, optional
        the binning in the height direction. The default is 1.

    Returns
    -------
    None.

    """
    ########################### setting binning ###############################
    possible_bin_values = [1, 2, 4, 8, 16]
    if binningX in possible_bin_values:
        print('ok')
        cam.set_binning_horizontal(binningX) 
        print('binning X set to : ' + str(cam.get_binning_horizontal()))
    else:
        print('binningX not accept, it must be set to : 1, 2, 4, 8 or 16')
    if binningY in possible_bin_values:
        cam.set_binning_vertical(binningY) 
        print('binning Y set to : ' + str(cam.get_binning_vertical()))
    else:
        print('binningY not accept, it must be set to : 1, 2, 4, 8 or 16')
    ########################### setting ROI ###################################
    # width = cam.get_width()
    width_max = cam.get_width_maximum()
    width_inc = cam.get_width_increment() 

    # height = cam.get_height()
    height_max = cam.get_height_maximum()
    height_inc = cam.get_height_increment() 

    # offsetX = cam.get_offsetX()
    # offsetX_min = cam.get_offsetX_minimum()
    # offsetX_max = cam.get_offsetX_maximum()
    offsetX_inc = cam.get_offsetX_increment() 

    # offsetY = cam.get_offsetY()
    # offsetY_min = cam.get_offsetY_minimum()
    # offsetY_max = cam.get_offsetY_maximum()
    offsetY_inc = cam.get_offsetY_increment() 

    width_acc = width#round(round(width / width_inc) * width_inc / binningX)
    height_acc = height# round(round(height / height_inc) * height_inc / binningY)
    offsetX_acc = round(offsetX / offsetX_inc) * offsetX_inc
    offsetY_acc = round(offsetY / offsetY_inc) * offsetY_inc

    # while True:
    #     if offsetX_acc < 0:
    #         print('problem, offsetX is negative, it is set to zero')
    #         offsetX_acc = 0
    #         break
    #     if width_acc + offsetX_acc > width_max:
    #         offsetX_acc = offsetX_acc - offsetX_inc
    #         print('offsetX + width higher than width max, offsetX decrease')
    #         if offsetX_acc < 0:
    #             print('problem, offsetX is negative, it is set to zero')
    #             offsetX_acc = 0
    #             break
    #     else:
    #         break

    # while True:
    #     if offsetY_acc < 0:
    #         print('problem, offsetY is negative, it is set to zero')
    #         offsetY_acc = 0
    #         break
    #     if height_acc + offsetY_acc > height_max:
    #         offsetY_acc = offsetY_acc - offsetY_inc
    #         print('offsetY + height higher than height max, offsetY decrease')
    #         if offsetY_acc < 0:
    #             print('problem, offsetY is negative, it is set to zero')
    #             offsetY_acc = 0
    #             break
    #     else:
    #         break

    cam.set_width(width_acc) 
    cam.set_height(height_acc) 
    cam.set_offsetX(offsetX_acc) 
    cam.set_offsetY(offsetY_acc) 
    
    width_get = cam.get_width()
    height_get = cam.get_height()
    offsetX_get = cam.get_offsetX()
    offsetY_get = cam.get_offsetY()
    
    print('width set to    : ' + str(width_get))
    print('height set to   : ' + str(height_get))
    print('offset X set to : ' + str(offsetX_get))
    print('offset Y set to : ' + str(offsetY_get))
    ####################### set sensor bit depth ##############################
    cam.set_sensor_bit_depth('XI_BPP_10')
    ###################### set output data format #############################
    cam.set_output_bit_depth('XI_BPP_10') 
    #################### setting the exposure timre ###########################
    # NB: the exposure time must bean interger in µs wheras it is enqueried in ms as a float
    exposure_time = round(expos_time * 1000)
    if exposure_time < cam.get_exposure_minimum():
        exposure_time = cam.get_exposure_minimum()
        print('the exposure time is below the minimum value, it is set to ' + str(cam.get_exposure_minimum()))
    if exposure_time > cam.get_exposure_maximum():
        exposure_time = cam.get_exposure_maximum()
        print('the exposure time is above the maximum value, it is set to ' + str(cam.get_exposure_maximum()))
        
    cam.set_exposure(exposure_time)
    exposure_time_get = cam.get_exposure()
    print('exposure time set to : ' + str(exposure_time_get / 1000) + ' ms')
    ######################### get the frame rate ##############################
    current_frame_rate = cam.get_framerate()
    print('frame rate is deduced to : ' + str(current_frame_rate))
    print('!!! warning, at this moment, the frame rate cannot be set')
    # cam.set_framerate(frame_rate)
    # current_frame_rate = cam.get_framerate()
    # print('new frame rate set to : ' + str(current_frame_rate))
    ########### set the image data format to 8 bit to display it ##############
    if cam.get_device_name().decode(encoding) == 'CB013CG-LX-X8G3':
        cam.set_imgdataformat('XI_RGB24')
        # print('image data format set to : ' + cam.get_imgdataformat() + ' for the color camera')
    elif cam.get_device_name().decode(encoding) == 'CB013MG-LX-X8G3':
        cam.set_imgdataformat('XI_RAW8')
        # print('image data format set to : ' + cam.get_imgdataformat() + ' for the monochrome camera')
    ####################### setting the data rate #############################
    cam.set_limit_bandwidth_mode('XI_ON')
    CAMERAS_ON_SAME_CONTROLLER = cameras_nbr
    #set interface data rate
    interface_data_rate = cam.get_limit_bandwidth()
    camera_data_rate = int(interface_data_rate / CAMERAS_ON_SAME_CONTROLLER)

    # get min and max data rate:
    min_data_rate_cam = cam.get_limit_bandwidth_minimum()
    max_data_rate_cam = cam.get_limit_bandwidth_maximum()
    
    if camera_data_rate < min_data_rate_cam:
        camera_data_rate = min_data_rate_cam
        print('camera_data_rate is below the minimum value, it is set to its miminum value')        
    elif camera_data_rate > max_data_rate_cam:
        camera_data_rate = max_data_rate_cam
        print('camera_data_rate is above the maximum value, it is set to its maxinum value')
    # ici, je prend la valeur max que je divise par deux, à améliorer
    camera_data_rate = int(max_data_rate_cam / CAMERAS_ON_SAME_CONTROLLER)
   
    cam.set_limit_bandwidth(camera_data_rate)
    ########################## set buffer #####################################
    cam.set_buffer_policy('XI_BP_SAFE')
    cam.set_acq_buffer_size(int(cam.get_acq_buffer_size_maximum()/4)) # divide by 4 because if higher, we lose triggers, to set max, you need too wait 2.6s between star_acquisition and receive the first trig (DMD.run), to set max/2 => wait 1.5s, max/4 => wait 1s
    print('buffer size = ' + str(cam.get_acq_buffer_size()))
    cam.set_buffers_queue_size(cam.get_buffers_queue_size_maximum()) 
    buffers_queue_size = cam.get_buffers_queue_size()
    print('buffers queue size  = ' + str(buffers_queue_size ))
    ########################## setting gain ###################################
    gain_min = cam.get_gain_minimum()
    gain_max = cam.get_gain_maximum()
    # gain_inc = cam.get_gain_increment()
    if gain < gain_min:
        gain = 0
        print('Warning, gain is below the minimum value, it is set to ' + str(round(gain_min, 2)) + ' dB')        
    elif gain > gain_max:
        gain = gain_max
        print('Warning, gain is above the maximum value, it is set to ' + str(round(gain_max, 2)) + ' dB')       
                    
    cam.set_gain(gain)
    gain_get = cam.get_gain()
    print('the gain is set to : ' + str(round(gain_get, 2)) + ' dB')
    ######################## setting white balance ############################
    if cam.get_device_name().decode(encoding) == 'CB013CG-LX-X8G3':
        # auto_wb = cam.is_auto_wb() 
        if auto_wb == True:
            cam.enable_auto_wb()
        elif auto_wb == False:
            cam.disable_auto_wb() 
    
        wb_kr = cam.get_wb_kr()
        wb_kg = cam.get_wb_kg()
        wb_kb = cam.get_wb_kb()
    ######################### set the GammaY ##################################
    gammaY_min = cam.get_gammaY_minimum()
    gammaY_max = cam.get_gammaY_maximum()
    if gammaY < gammaY_min:
        gammaY = gammaY_min
        print('gammaY is below the minimum, it is set to : ' + str(round(gammaY_min, 2)))
    elif gammaY > gammaY_max:
        gammaY = gammaY_max
        print('gammaY is above the maximum, it is set to : ' + str(round(gammaY_max, 2)))
        
    cam.set_gammaY(gammaY)    
    ########### set the image data format to 10 bit to save it ################
    if cam.get_device_name().decode(encoding) == 'CB013CG-LX-X8G3':
        cam.set_imgdataformat('XI_RGB48')
        # print('image data format set to : ' + cam.get_imgdataformat() + ' for the color camera')
    elif cam.get_device_name().decode(encoding) == 'CB013MG-LX-X8G3':
        cam.set_imgdataformat('XI_RAW16')
        # print('the exposure time is above the maximum value, it is set to ' + str(cam.get_exposure_maximum()))
        
    ################# acquire waiting an internal trigger #####################
    gpi_mode = 'XI_GPI_TRIGGER'
    cam.set_gpi_mode(gpi_mode)
    trigger_source = 'XI_TRG_EDGE_RISING'
    cam.set_trigger_source(trigger_source)
    trigger_selector = 'XI_TRG_SEL_FRAME_START'
    cam.set_trigger_selector(trigger_selector)
    
    return cam_Parameters(cam = cam)
        

def snapshot_cam(cam, data_format: int = 8, binX: int = 1, binY: int = 1, disp_bin_effect: bool = False):
    """
    take a snapshot of the camera
    
    Parameters:
    -----------
        cam (obj): 
            a object to drive the Ximea camera
        data_format (int):
            Format of image data returned by function xiGetImage in depth bit
        binX (int):
            binning in the width direction
        binY (int):
            binning in the height direction
        disp_bin_effect (bool):
            display graph and SNR measurement cause by the binning
    Returns
    -------
    None.

    """
    def rebin(arr, new_shape):
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])
        return arr.reshape(shape).mean(-1).mean(1)
    
    ########### set the image data format to 8 bit to display it ##############
    if data_format == 8:
        if cam.get_device_name().decode(encoding) == 'CB013CG-LX-X8G3':
            cam.set_imgdataformat('XI_RGB24')
            print('image data format set to : ' + cam.get_imgdataformat() + ' for the color camera')
        elif cam.get_device_name().decode(encoding) == 'CB013MG-LX-X8G3':
            cam.set_imgdataformat('XI_RAW8')
            print('image data format set to : ' + cam.get_imgdataformat() + ' for the monochrome camera')
            
    image_bit_depth_str = cam.get_image_data_bit_depth()
    image_bit_depth = int(image_bit_depth_str[7:])
    print('data output bit depth = ' + str(image_bit_depth))
    ##################### create instance for cameras #########################
    img = xiapi.Image()
    ####################### start data acquisition ############################
    print('Starting data acquisition...\n')
    cam.start_acquisition()
    ############## get data and pass them from cameras to img #################
    cam.get_image(img)
    ################### get image data as numpy array #########################
    data = img.get_image_data_numpy(invert_rgb_order = True)
    ####################### stop data acquisition #############################
    print(cam.arm + ' camera: Stopping acquisition...')
    cam.stop_acquisition()
    ######################## check saturation #################################
    data_max = np.max(data)
    if (data_max >= 255 and image_bit_depth == 8) or (data_max >= 1023 and image_bit_depth == 10) or (data_max >= 65535 and image_bit_depth == 16):
        print('!!!!!!!!!! Warning, saturation detected !!!!!!!!!!!!')
    ######################## apply binning ####################################
    if binX != 1 and binY !=1:
        data_bin = rebin(data, [round(cam.get_height()/binY), round(cam.get_width()/binX)])
    else:
        data_bin = data
    ########################## print snapshot #################################
    plt.figure()
    plt.imshow(data)
    plt.colorbar()
    plt.title(cam.arm + ' cam')
    plt.xlabel('X (Width)')
    plt.ylabel('Y (Height)')
    
    Sig = np.max(data[round(data.shape[0]/2 - 100):round(data.shape[0]/2 + 100), round(data.shape[1]/2 - 100):round(data.shape[1]/2 + 100)])
    noise = np.std(data[round(50/(data.shape[0]/cam.get_height())):round(150/(data.shape[0]/cam.get_height())), round(data.shape[1]/2 - 100):round(data.shape[1]/2 + 100)].flatten())
    maxi = np.max(data)
    print('Original :')
    print('     max   = ' + str(maxi))
    print('     Sig   = ' + str(Sig))
    print('     Noise = ' + str(noise))
    print('     SNR   = ' + str(Sig/noise))
    
    if disp_bin_effect == True:
        plt.figure()
        plt.imshow(data_bin)
        plt.colorbar()
        plt.title(cam.arm + ' cam / binning')
        plt.xlabel('X (Width)')
        plt.ylabel('Y (Height)')
        
        Sig_bin = np.max(data_bin[round(data_bin.shape[0]/2 - 100/binY):round(data_bin.shape[0]/2 + 100/binY), round(data_bin.shape[1]/2 - 100/binX):round(data_bin.shape[1]/2 + 100/binX)])
        noise_bin = np.std(data_bin[round(50/binY):round(150/binY), round(data_bin.shape[1]/2 - 100/binX):round(data_bin.shape[1]/2 + 100/binX)].flatten())
        print('After binning :')
        print('     Sig = ' + str(Sig_bin))
        print('     Noise = ' + str(noise_bin))
        print('     SNR = ' + str(Sig_bin/noise_bin))
    ########### set the image data format to 10 bit to save it ################
    if data_format == 8:
        if cam.get_device_name().decode(encoding) == 'CB013CG-LX-X8G3':
            cam.set_imgdataformat('XI_RGB48')
            print('image data format set to : ' + cam.get_imgdataformat() + ' for the color camera')
        elif cam.get_device_name().decode(encoding) == 'CB013MG-LX-X8G3':
            cam.set_imgdataformat('XI_RAW16')
            print('image data format set to : ' + cam.get_imgdataformat() + ' for the monochrome camera')
    
    return data    


def display_cam(cam):
    """
    Continuous image display of a camera
    
    Parameters:
    -----------
        cam (obj): 
            a object to drive the Ximea camera
            
    Returns:
    -------
        None
    """
    
    try:
    
        import cv2
        # Creating a cv2 window
        window_name = "Camera of the Spatial Arm"
        cv2.namedWindow(window_name) 
        
        # Create a function 'nothing' for creating trackbar 
        def nothing(x): 
            pass
        
        ####### set the image data format to 8 bit to display it ##############
        if cam.get_device_name().decode(encoding) == 'CB013CG-LX-X8G3':
            cam.set_imgdataformat('XI_RGB24')
            print('image data format set to : ' + cam.get_imgdataformat() + ' for the color camera')
        elif cam.get_device_name().decode(encoding) == 'CB013MG-LX-X8G3':
            cam.set_imgdataformat('XI_RAW8')
            print('image data format set to : ' + cam.get_imgdataformat() + ' for the monochrome camera')
            
        image_bit_depth_str = cam.get_image_data_bit_depth()
        image_bit_depth = int(image_bit_depth_str[7:])
        # waiting time inside the loop of the display of the window
        t1 = cam.get_exposure()/1000 # (ms)
        t2 = 1/cam.get_framerate()/1000
        t_wait = max(t1, t2)/1000
        print('wait time = ' + str(t_wait))
        
        #create instance of Image to store image data and metadata
        img = xiapi.Image()
        
        min_exposure_time = int(cam.get_exposure_minimum())
        max_exposure_time = int(cam.get_exposure_maximum())
        current_exposure_time = cam.get_exposure()
        
        gain = cam.get_gain()
        gain_min = cam.get_gain_minimum()
        gain_max = cam.get_gain_maximum()
    
        first_passage = True
        
        #start data acquisition
        print('Start acquisition...\n')
        cam.start_acquisition()
        
        first_passage2 = True
        data_center_old = 0
        maxii = 0
        while True:
            time.sleep(t_wait) # Sleep for 1 seconds
            
            #get data and pass them from cameras to img
            cam.get_image(img)
            
            data = img.get_image_data_numpy()
            
            maxi = np.max(data)
            if maxi != maxii:
                print("max = " + str(maxi))
                maxii = maxi
            
            if np.max(data) == 255 and first_passage2 == True:
                print('saturation detected')
                first_passage2 = False
            elif np.max(data) < 255 and first_passage2 == False:
                print('No more saturation')
                first_passage2 = True
            
            if first_passage == True:
                maxi = np.max(data)
                print('maxi = ' + str(maxi))
                print('press "q" to exit')
                # Creating trackbars for color change 
                cv2.createTrackbar('Brightness', window_name, maxi, 510, nothing) 

                cv2.createTrackbar('Exp time', window_name, int(current_exposure_time), 50000, nothing) 

                cv2.createTrackbar('Gain', window_name, int(gain), int(gain_max), nothing) 
                first_passage = False
                # print('data = ' + str(data[550,1100,:]))
                last_gain = cam.get_gain()
            
            # Get current positions of trackbar 
            brightness = cv2.getTrackbarPos('Brightness', window_name) 
            
            # Tune exposure time
            exposure_time = cv2.getTrackbarPos('Exp time', window_name)             
            if exposure_time < min_exposure_time:
                exposure_time = min_exposure_time                
            elif exposure_time > max_exposure_time:
                exposure_time = max_exposure_time
            
            cam.set_exposure(exposure_time)
            
            
            # Tune gain
            gain = cv2.getTrackbarPos('Gain', window_name) 
            if gain != last_gain:
                if gain < gain_min:
                    gain = gain_min
                elif gain > gain_max:
                    gain = gain_max
                    
                cam.set_gain(gain)
                last_gain = gain
            
        
            data = data.astype(np.float64)
            data2 = data*brightness/maxi
            data3 = data2.astype(np.uint8)
            
            # if cam.get_device_name().decode(encoding) == 'CB013MG-LX-X8G3':
            #     data_center = data[432, 640]
            #     if data_center != data_center_old:
            #         print('data center = ' + str(data_center))
            #     data_center_old = data_center
            
            # if image_bit_depth == 8:
            #     data3 = data2.astype(np.uint8)
            # elif image_bit_depth == 16:
            #     data3 = data2.astype(np.uint16)
            # elif image_bit_depth == 32:
            #     data3 = data2.astype(np.uint32)

            cv2.imshow(window_name, data3)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow(window_name)
                #stop data acquisition
                print('cam: Stopping acquisition...')
                cam.stop_acquisition()
                
                ########### set the image data format to 10 bit to save it ##############
                if cam.get_device_name().decode(encoding) == 'CB013CG-LX-X8G3':
                    cam.set_imgdataformat('XI_RGB48')
                    print('image data format set to : ' + cam.get_imgdataformat() + ' for the color camera')
                elif cam.get_device_name().decode(encoding) == 'CB013MG-LX-X8G3':
                    cam.set_imgdataformat('XI_RAW16')
                    print('image data format set to : ' + cam.get_imgdataformat() + ' for the B & W camera')
                    
                break
        
    except:
        cv2.destroyWindow(window_name)
        cam.stop_acquisition()
        print('try function encoutered a exception')
        ########### set the image data format to 10 bit to save it ##############
        if cam.get_device_name().decode(encoding) == 'CB013CG-LX-X8G3':
            cam.set_imgdataformat('XI_RGB48')
            print('image data format set to : ' + cam.get_imgdataformat() + ' for the color camera')
        elif cam.get_device_name().decode(encoding) == 'CB013MG-LX-X8G3':
            cam.set_imgdataformat('XI_RAW16')
            print('the exposure time is above the maximum value, it is set to ' + str(cam.get_exposure_maximum()))
        
        
def counter_trigger(cam):
    """
    Arg:
        cam (obj): 
            a object to drive the Ximea camera
    Returns:
        a tuple containing counter of the trigger skipped and received
    """
    cam.set_counter_selector('XI_CNT_SEL_TRANSPORT_SKIPPED_FRAMES')
    transport_skipped_trig = cam.get_counter_value()
    cam.set_counter_selector('XI_CNT_SEL_API_SKIPPED_FRAMES')
    api_skipped_trig = cam.get_counter_value()
    cam.set_counter_selector('XI_CNT_SEL_TRANSPORT_TRANSFERRED_FRAMES')
    transported_frames = cam.get_counter_value()
    
    return [transport_skipped_trig, api_skipped_trig, transported_frames]        
        
        
        
        
        
        
        
        
        
        
        
        