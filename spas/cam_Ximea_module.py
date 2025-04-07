# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:48:30 2025

@author: mahieu
"""
encoding = 'utf-8'

from ximea import xiapi
import time
import numpy as np

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

def disconnect_cam(self):
    """
    disconnect the camera
    
    Parameters:
    -----------
        self (obj): 
            a object to drive the Ximea camera

    Returns
    -------
    None.

    """
    
    # cam_name = self.get_device_name().decode(encoding)
    self.close_device()
    if self.arm == 'spatial':
        print('spatial  camera disconnected')
    elif self.arm == 'spectral':
        print('spectral camera disconnected')
    
        
def setup_cam(self, cameras_nbr: int = 2, expos_time: float = 1, gain: float = 0, auto_wb: bool = True,
              width: int = 1280, height: int = 864, offsetX: int = 0, offsetY: int = 0):
    """
    setup the Ximea camera

    Parameters
    ----------
    self (obj): 
        a object to drive the Ximea camera
    cameras_nbr : int, optional
        the number of camera connected on the same controller. The default is 2.
    expos_time : float, optional
        the exposure time in ms. The default is 1.
    gain : float, optional
        DESCRIPTION. The default is 0.
    auto_wb : bool, optional
        DESCRIPTION. The default is True.
    width : int, optional
        DESCRIPTION. The default is 1280.
    height : int, optional
        DESCRIPTION. The default is 864.
    offsetX : int, optional
        DESCRIPTION. The default is 0.
    offsetY : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    ####################### setting the data rate #############################
    CAMERAS_ON_SAME_CONTROLLER = cameras_nbr
    #set interface data rate
    interface_data_rate = self.get_limit_bandwidth()
    camera_data_rate = int(interface_data_rate / CAMERAS_ON_SAME_CONTROLLER)

    # get min and max data rate:
    min_data_rate_cam_spat = self.get_limit_bandwidth_minimum()
    max_data_rate_cam_spat = self.get_limit_bandwidth_maximum()
    
    if camera_data_rate < min_data_rate_cam_spat:
        camera_data_rate = min_data_rate_cam_spat
        print('camera_data_rate is below the minimum value, it is set to its miminum value')
        
    if camera_data_rate > max_data_rate_cam_spat:
        camera_data_rate = max_data_rate_cam_spat
        print('camera_data_rate is above the maximum value, it is set to its maxinum value')

    camera_data_rate = int(max_data_rate_cam_spat / CAMERAS_ON_SAME_CONTROLLER)
   
    self.set_limit_bandwidth(camera_data_rate)
    ########### set the image data format for the color camera ################
    if self.get_device_name().decode(encoding) == 'CB013CG-LX-X8G3':
        self.set_imgdataformat('XI_RGB24')
    #################### setting the exposure timre ###########################
    # NB: the exposure time must bean interger in µs wheras it is enqueried in ms as a float
    exposure_time = round(expos_time * 1000)
    if exposure_time < self.get_exposure_minimum():
        exposure_time = self.get_exposure_minimum()
        print('the exposure time is below the minimum value, it is set to ' + str(self.get_exposure_minimum()))
    if exposure_time > self.get_exposure_maximum():
        exposure_time = self.get_exposure_maximum()
        print('the exposure time is above the maximum value, it is set to ' + str(self.get_exposure_maximum()))
        
    self.set_exposure(exposure_time)
    exposure_time_get = self.get_exposure()
    print('exposure time set to : ' + str(exposure_time_get / 1000) + ' ms')
    ########################## setting gain ###################################
    gain_min = self.get_gain_minimum()
    gain_max = self.get_gain_maximum()
    # gain_inc = self.get_gain_increment()
    if gain < gain_min:
        gain = 0
        print('Warning, gain is below the minimum value, it is set to ' + str(gain_min))        
    elif gain > gain_max:
        gain = gain_max
        print('Warning, gain is above the maximum value, it is set to ' + str(gain_max))       
                    
    self.set_gain(gain)
    gain_get = self.get_gain()
    print('the gain is set to : ' + str(gain_get))
    ######################## setting white balance ############################
    # auto_wb = self.is_auto_wb() 
    if auto_wb == True:
        self.enable_auto_wb()
    elif auto_wb == False:
        self.disable_auto_wb() 

    wb_kr = self.get_wb_kr()
    wb_kg = self.get_wb_kg()
    wb_kb = self.get_wb_kb()

    ########################### setting ROI ###################################
    # width = self.get_width()
    width_max = self.get_width_maximum()
    width_inc = self.get_width_increment() 

    # height = self.get_height()
    height_max = self.get_height_maximum()
    height_inc = self.get_height_increment() 

    # offsetX = self.get_offsetX()
    # offsetX_min = self.get_offsetX_minimum()
    # offsetX_max = self.get_offsetX_maximum()
    offsetX_inc = self.get_offsetX_increment() 

    # offsetY = self.get_offsetY()
    # offsetY_min = self.get_offsetY_minimum()
    # offsetY_max = self.get_offsetY_maximum()
    offsetY_inc = self.get_offsetY_increment() 

    width_acc = round(width / width_inc) * width_inc
    height_acc = round(height / height_inc) * height_inc
    offsetX_acc = round(offsetX / offsetX_inc) * offsetX_inc
    offsetY_acc = round(offsetY / offsetY_inc) * offsetY_inc

    while True:
        if offsetX_acc < 0:
            print('problem, offsetX is negative, it is set to zero')
            offsetX_acc = 0
            break
        if width_acc + offsetX_acc > width_max:
            offsetX_acc = offsetX_acc - offsetX_inc
            print('offsetX + width higher than width max, offsetX decrease')
            if offsetX_acc < 0:
                print('problem, offsetX is negative, it is set to zero')
                offsetX_acc = 0
                break
        else:
            break

    while True:
        if offsetY_acc < 0:
            print('problem, offsetY is negative, it is set to zero')
            offsetY_acc = 0
            break
        if height_acc + offsetY_acc > height_max:
            offsetY_acc = offsetY_acc - offsetY_inc
            print('offsetY + height higher than height max, offsetY decrease')
            if offsetY_acc < 0:
                print('problem, offsetY is negative, it is set to zero')
                offsetY_acc = 0
                break
        else:
            break

    self.set_width(width_acc) 
    self.set_height(height_acc) 
    self.set_offsetX(offsetX_acc) 
    self.set_offsetY(offsetY_acc) 
    
    width_get = self.get_width()
    height_get = self.get_height()
    offsetX_get = self.get_offsetX()
    offsetY_get = self.get_offsetY()
    
    print('width set to : ' + str(width_get))
    print('height set to : ' + str(height_get))
    print('offset X set to : ' + str(offsetX_get))
    print('offset Y set to : ' + str(offsetY_get))


def display_cam(self):
    """
    Continuous image display of a camera
    
    Parameters:
    -----------
        self (obj): 
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
        
        # waiting time inside the loop of the display of the window
        t1 = self.get_exposure()/1000 # (ms)
        t2 = 1/self.get_framerate()/1000
        t_wait = max(t1, t2)/1000
        print('wait time = ' + str(t_wait))
        
        #create instance of Image to store image data and metadata
        img_spat = xiapi.Image()
        
        
        
        min_exposure_time = int(self.get_exposure_minimum())
        max_exposure_time = int(self.get_exposure_maximum())
        current_exposure_time = self.get_exposure()
        
        gain = self.get_gain()
        gain_min = self.get_gain_minimum()
        gain_max = self.get_gain_maximum()
    
        first_passage = True
        
        #start data acquisition
        print('Start acquisition...\n')
        self.start_acquisition()
        
        first_passage2 = True
        
        while True:
            time.sleep(t_wait) # Sleep for 1 seconds
            
            #get data and pass them from cameras to img
            self.get_image(img_spat)
            
            data_spat = img_spat.get_image_data_numpy()
            
            if np.max(data_spat) == 255 and first_passage2 == True:
                print('saturation detected')
                first_passage2 = False
            elif np.max(data_spat) < 255 and first_passage2 == False:
                print('No more saturation')
                first_passage2 = True
            
            if first_passage == True:
                maxi = np.max(data_spat)
                print('maxi = ' + str(maxi))
                print('press "q" to exit')
                # Creating trackbars for color change 
                cv2.createTrackbar('Brightness', window_name, maxi, 510, nothing) 
                cv2.createTrackbar('Exp time', window_name, int(current_exposure_time), 50000, nothing) 
                cv2.createTrackbar('Gain', window_name, int(gain), int(gain_max), nothing) 
                first_passage = False
            
            # Get current positions of trackbar 
            brightness = cv2.getTrackbarPos('Brightness', window_name) 
            
            # Tune exposure time
            exposure_time = cv2.getTrackbarPos('Exp time', window_name)             
            if exposure_time < min_exposure_time:
                exposure_time = min_exposure_time                
            elif exposure_time > max_exposure_time:
                exposure_time = max_exposure_time
            
            self.set_exposure(exposure_time)
            
            
            # Tune gain
            gain = cv2.getTrackbarPos('Gain', window_name) 
            if gain < gain_min:
                gain = gain_min
            elif gain > gain_max:
                gain = gain_max
                
            self.set_gain(gain)
            
        
            data_spat = data_spat.astype(np.float64)
            data_spat2 = data_spat*brightness/maxi
            data_spat3 = data_spat2.astype(np.uint8)
            #*brightness/maxi
            cv2.imshow(window_name, data_spat3)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow(window_name)
                #stop data acquisition
                print('cam: Stopping acquisition...')
                self.stop_acquisition()
                break
        
    except:
        cv2.destroyWindow(window_name)
        self.stop_acquisition()
        print('try function encoutered a exception')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        