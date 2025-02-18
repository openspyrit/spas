# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:10:13 2025

@author: chiliaeva

init
piloting
disconnect
"""

import json
from ximea import xiapi 
from dataclasses import dataclass, InitVar, field
from typing import Optional, Union, List, Tuple, Optional
from dataclasses_json import dataclass_json
import numpy as np



'''   
EXAMPLE : 
    
cam = xiapi.Camera()

cam.open_device_by_SN('41305651') # open by serial number

print('Opening first camera...')
cam.open_device()


# Settings
cam.set_exposure(10000)


#create instance of Image to store image data and metadata
img = xiapi.Image()


#start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()




for i in range(10):
    # get data and pass them from camera to img
    cam.get_image(img)
    
    # get raw data from camera
    #for Python2.x function returns string
    #for Python3.x function returns bytes
    data_raw = img.get_image_data_raw()
    
    #transform data to list
    data = list(data_raw)
    
    #print image data and metadata
    print('Image number: ' + str(i))
    print('Image width (pixels):  ' + str(img.width))
    print('Image height (pixels): ' + str(img.height))
    print('First 10 pixels: ' + str(data[:10]))
    print('\n')    
    
    
    
#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

#stop communication
cam.close_device()

print('Done.')
'''
    
#########################################################################################################################################################
# Ximea version
##########################################################################################################################################################


def _init_CAM():
    """
    Initialize and connect to the Ximea camera.
    
    Returns:
        CAM: a structure containing the parameters of the IDS camera
    """   
    
    camPar = CAM(hCam = ueye.HIDS(0),
                 sInfo = ueye.SENSORINFO(),
                 cInfo = ueye.CAMINFO(),
                 nBitsPerPixel = ueye.INT(8),
                 m_nColorMode = ueye.INT(),
                 bytes_per_pixel = int( ueye.INT(8)/ 8),
                 rectAOI = ueye.IS_RECT(),
                 pcImageMemory = ueye.c_mem_p(),
                 MemID = ueye.int(),
                 pitch = ueye.INT(),
                 fps = float(), 
                 gain = int(), 
                 gainBoost = str(),
                 gamma = float(), 
                 exposureTime = float(), 
                 blackLevel = int(),
                 camActivated = bool(),
                 pixelClock = ueye.uint(),
                 bandwidth = float(),
                 Memory = bool(),
                 Exit = int(),
                 vidFormat = str(),
                 gate_period = int(),
                 trigger_mode = str(),
                 avi = ueye.int(),
                 punFileID = ueye.c_uint(),
                 timeout = int(),
                 time_array = [],
                 int_time_spect = float(), # unused ? 
                 black_pattern_num = int(), # unused ? 
                 insert_patterns = bool(), # unused ? 
                 acq_mode = str(), # unused ? 
                 )          





































    
##############################################################################################################################################################
# IDS version of the program 
############################################################################################################################################################## 

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
    if dll_pyueye_installed:
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
        
    
#####################################################################################################################    
    
    
    
    
def _init_CAM():
    """
    Initialize and connect to the IDS camera.
    
    Returns:
        CAM: a structure containing the parameters of the IDS camera
    """    
    camPar = CAM(hCam = ueye.HIDS(0),
                 sInfo = ueye.SENSORINFO(),
                 cInfo = ueye.CAMINFO(),
                 nBitsPerPixel = ueye.INT(8),
                 m_nColorMode = ueye.INT(),
                 bytes_per_pixel = int( ueye.INT(8)/ 8),
                 rectAOI = ueye.IS_RECT(),
                 pcImageMemory = ueye.c_mem_p(),
                 MemID = ueye.int(),
                 pitch = ueye.INT(),
                 fps = float(), 
                 gain = int(), 
                 gainBoost = str(),
                 gamma = float(), 
                 exposureTime = float(), 
                 blackLevel = int(),
                 camActivated = bool(),
                 pixelClock = ueye.uint(),
                 bandwidth = float(),
                 Memory = bool(),
                 Exit = int(),
                 vidFormat = str(),
                 gate_period = int(),
                 trigger_mode = str(),
                 avi = ueye.int(),
                 punFileID = ueye.c_uint(),
                 timeout = int(),
                 time_array = [],
                 int_time_spect = float(), # unused ? 
                 black_pattern_num = int(), # unused ? 
                 insert_patterns = bool(), # unused ? 
                 acq_mode = str(), # unused ? 
                 )          

    # # Camera Initialization ---
    # print("START Initialization of the IDS camera")
    ### Starts the driver and establishes the connection to the camera
    nRet = ueye.is_InitCamera(camPar.hCam, None)
    if nRet != ueye.IS_SUCCESS:
        print("is_InitCamera ERROR")

    ### Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
    nRet = ueye.is_GetCameraInfo(camPar.hCam, camPar.cInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetCameraInfo ERROR")

    ### You can query additional information about the sensor type used in the camera
    nRet = ueye.is_GetSensorInfo(camPar.hCam, camPar.sInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetSensorInfo ERROR")

    ### set camera parameters to default values
    nRet = ueye.is_ResetToDefault(camPar.hCam)
    if nRet != ueye.IS_SUCCESS:
        print("is_ResetToDefault ERROR")

    ### Set display mode to DIB
    nRet = ueye.is_SetDisplayMode(camPar.hCam, ueye.IS_SET_DM_DIB)

    ### Set the right color mode
    if int.from_bytes(camPar.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
        # setup the color depth to the current windows setting
        ueye.is_GetColorDepth(camPar.hCam, camPar.nBitsPerPixel, camPar.m_nColorMode)
        camPar.bytes_per_pixel = int(camPar.nBitsPerPixel / 8)
        # print("IS_COLORMODE_BAYER: ", )
        # print("\tm_nColorMode: \t\t", m_nColorMode)
        # print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        # print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        # print()

    elif int.from_bytes(camPar.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
        # for color camera models use RGB32 mode
        camPar.m_nColorMode = ueye.IS_CM_BGRA8_PACKED
        camPar.nBitsPerPixel = ueye.INT(32)
        camPar.bytes_per_pixel = int(camPar.nBitsPerPixel / 8)
        # print("IS_COLORMODE_CBYCRY: ", )
        # print("\tm_nColorMode: \t\t", m_nColorMode)
        # print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        # print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        # print()

    elif int.from_bytes(camPar.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
        # for color camera models use RGB32 mode
        camPar.m_nColorMode = ueye.IS_CM_MONO8
        camPar.nBitsPerPixel = ueye.INT(8)
        camPar.bytes_per_pixel = int(camPar.nBitsPerPixel / 8)
        # print("IS_COLORMODE_MONOCHROME: ", )
        # print("\tm_nColorMode: \t\t", m_nColorMode)
        # print("\tnBitsPerPixel: \t\t", nBitsPerPixel)
        # print("\tbytes_per_pixel: \t\t", bytes_per_pixel)
        # print()

    else:
        # for monochrome camera models use Y8 mode
        camPar.m_nColorMode = ueye.IS_CM_MONO8
        camPar.nBitsPerPixel = ueye.INT(8)
        camPar.bytes_per_pixel = int(camPar.nBitsPerPixel / 8)
        # print("else")
    
    ### Get the AOI (Area Of Interest)
    sizeofrectAOI = ueye.c_uint(4*4)
    nRet = ueye.is_AOI(camPar.hCam, ueye.IS_AOI_IMAGE_GET_AOI, camPar.rectAOI, sizeofrectAOI)
    if nRet != ueye.IS_SUCCESS:
        print("AOI getting ERROR")
    
    camPar.camActivated = 0
    
    # Get current pixel clock
    getpixelclock = ueye.UINT(0)
    check_ueye(ueye.is_PixelClock, camPar.hCam, ueye.PIXELCLOCK_CMD.IS_PIXELCLOCK_CMD_GET, getpixelclock,
               ueye.sizeof(getpixelclock))
    
    camPar.pixelClock = getpixelclock
    # print('pixel clock = ' + str(getpixelclock) + ' MHz')
    
    # get the bandwidth (in MByte/s)
    camPar.bandwidth = ueye.is_GetUsedBandwidth(camPar.hCam)
    # print('Bandwidth = ' + str(camPar.bandwidth) + ' MB/s')
    
    
    # width = rectAOI.s32Width
    # height = rectAOI.s32Height

    # Prints out some information about the camera and the sensor
    # print("Camera model:\t\t", sInfo.strSensorName.decode('utf-8'))
    # print("Camera serial no.:\t", cInfo.SerNo.decode('utf-8'))
    # print("Maximum image width:\t", width)
    # print("Maximum image height:\t", height)
    # print()    
        
    # self.hCam = hCam
    # self.sInfo = sInfo
    # self.cInfo = cInfo
    # self.nBitsPerPixel = nBitsPerPixel
    # self.m_nColorMode = m_nColorMode
    # self.bytes_per_pixel = bytes_per_pixel
    # self.rectAOI = rectAOI
    
    camPar.Exit = 1
    
    print('IDS camera connected')
    
    return camPar    
    
    
 
    
    
def captureVid(camPar):
    """
    Allocate memory and begin video capture of the IDS camera

    Args:
        camPar : a structure containing the parameters of the IDS camera.

    Returns:
        camPar : a structure containing the parameters of the IDS camera.
    """
    camPar = stopCapt_DeallocMem_ExitCam(camPar)
    
    if camPar.Exit == 0:
        camPar = _init_CAM()
        camPar.Exit = 1
        
    
    ### Set the AOI
    sizeofrectAOI = ueye.c_uint(4*4)
    nRet = ueye.is_AOI(camPar.hCam, ueye.IS_AOI_IMAGE_SET_AOI, camPar.rectAOI, sizeofrectAOI)
    if nRet != ueye.IS_SUCCESS:
        print("AOI setting ERROR")

    width = camPar.rectAOI.s32Width
    height = camPar.rectAOI.s32Height

    ### Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
    nRet = ueye.is_AllocImageMem(camPar.hCam, width, height, camPar.nBitsPerPixel, camPar.pcImageMemory, camPar.MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_AllocImageMem ERROR")
    else:
        # Makes the specified image memory the active memory
        camPar.Memory = 1
        nRet = ueye.is_SetImageMem(camPar.hCam, camPar.pcImageMemory, camPar.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetImageMem ERROR")
        else:
            # Set the desired color mode
            nRet = ueye.is_SetColorMode(camPar.hCam, camPar.m_nColorMode)
    

    ### Activates the camera's live video mode (free run mode)
    nRet = ueye.is_CaptureVideo(camPar.hCam, ueye.IS_DONT_WAIT)
    if nRet != ueye.IS_SUCCESS:
        print("is_CaptureVideo ERROR")
        
        
    ### Enables the queue mode for existing image memory sequences
    nRet = ueye.is_InquireImageMem(camPar.hCam, camPar.pcImageMemory, camPar.MemID, width, height, camPar.nBitsPerPixel, camPar.pitch)
    if nRet != ueye.IS_SUCCESS:
        print("is_InquireImageMem ERROR")    

    camPar.camActivated = 1
    
    return camPar    
    
    
    

def setup_cam(camPar, pixelClock, fps, Gain, gain_boost, nGamma, ExposureTime, black_level):
    """
    Set and read the camera parameters
    
    Args:
        pixelClock = [118, 237 or 474] (MHz)
        fps: fps boundary => [1 - No Value] sup limit depend of image size (216 fps for 768x544 pixels for example)
        Gain: Gain boundary => [0 100]
        gain_boost: 'ON' set "ON" to activate gain boost, "OFF" to deactivate
        nGamma: Gamma boundary => [1 - 2.2]
        ExposureTime: Exposure time (ms) boundarye => [0.032 - 56.221] 
        black_level: Black Level boundary => [0 255] 
    
    returns:
        CAM: a structure containing the parameters of the IDS camera
    """
    # It is necessary to execute twice this code to take account the parameter modification
    for i in range(2): 
        ############################### Set Pixel Clock ###############################
        ### Get range of pixel clock, result : range = [118 474] MHz (Inc = 0)
        getpixelclock = ueye.UINT(0)
        newpixelclock = ueye.UINT(0)
        newpixelclock.value = pixelClock
        PixelClockRange = (ueye.int * 3)()
    
        # Get pixel clock range
        nRet = ueye.is_PixelClock(camPar.hCam, ueye.IS_PIXELCLOCK_CMD_GET_RANGE, PixelClockRange, ueye.sizeof(PixelClockRange))
        if nRet == ueye.IS_SUCCESS:
            nPixelClockMin = PixelClockRange[0]
            nPixelClockMax = PixelClockRange[1]
            nPixelClockInc = PixelClockRange[2]
    
        # Set pixel clock
        check_ueye(ueye.is_PixelClock, camPar.hCam, ueye.PIXELCLOCK_CMD.IS_PIXELCLOCK_CMD_SET, newpixelclock,
                   ueye.sizeof(newpixelclock))
        # Get current pixel clock
        check_ueye(ueye.is_PixelClock, camPar.hCam, ueye.PIXELCLOCK_CMD.IS_PIXELCLOCK_CMD_GET, getpixelclock,
                   ueye.sizeof(getpixelclock))
        
        camPar.pixelClock = getpixelclock.value
        if i == 1:
            print('            pixel clock = ' + str(getpixelclock) + ' MHz')
        if getpixelclock == 118:
            if i == 1:
                print('Pixel clcok blocked to 118 MHz, it is necessary to unplug the camera if not desired')
        # get the bandwidth (in MByte/s)
        camPar.bandwidth = ueye.is_GetUsedBandwidth(camPar.hCam)
        if i == 1:
            print('              Bandwidth = ' + str(camPar.bandwidth) + ' MB/s')
        ############################### Set FrameRate #################################
        ### Read current FrameRate
        dblFPS_init = ueye.c_double()
        nRet = ueye.is_GetFramesPerSecond(camPar.hCam, dblFPS_init)
        if nRet != ueye.IS_SUCCESS:
            print("FrameRate getting ERROR")
        else:
            dblFPS_eff = dblFPS_init
            if i == 1:
                print('            current FPS = '+str(round(dblFPS_init.value*100)/100) + ' fps')
            if fps < 1:
                fps = 1
                if i == 1:
                    print('FPS exceed lower limit >= 1')
                
            dblFPS = ueye.c_double(fps)  
            if (dblFPS.value < dblFPS_init.value-0.01) | (dblFPS.value > dblFPS_init.value+0.01):
                newFPS = ueye.c_double()
                nRet = ueye.is_SetFrameRate(camPar.hCam, dblFPS, newFPS)
                time.sleep(1)
                if nRet != ueye.IS_SUCCESS:
                    print("FrameRate setting ERROR")
                else:
                    if i == 1:
                        print('                new FPS = '+str(round(newFPS.value*100)/100) + ' fps')
                    ### Read again the effective FPS / depend of the image size, 17.7 fps is not possible with the entire image size (ie 2076x3088)
                    dblFPS_eff = ueye.c_double() 
                    nRet = ueye.is_GetFramesPerSecond(camPar.hCam, dblFPS_eff)
                    if nRet != ueye.IS_SUCCESS:
                        print("FrameRate getting ERROR")
                    else:       
                        if i == 1:
                            print('          effective FPS = '+str(round(dblFPS_eff.value*100)/100) + ' fps')
        ############################### Set GAIN ######################################
        #### Maximum gain is depending of the sensor. Convertion gain code to gain to limit values from 0 to 100
        # gain_code = gain * slope + b
        gain_max_code = 1450
        gain_min_code = 100
        gain_max = 100
        gain_min = 0
        slope = (gain_max_code-gain_min_code)/(gain_max-gain_min)
        b = gain_min_code
        #### Read gain setting
        current_gain_code = ueye.c_int()
        current_gain_code = ueye.is_SetHWGainFactor(camPar.hCam, ueye.IS_GET_MASTER_GAIN_FACTOR, current_gain_code)
        current_gain = round((current_gain_code-b)/slope) 
        
        if i == 1:
            print('           current GAIN = '+str(current_gain))    
        gain_eff = current_gain
        
        ### Set new gain value
        gain = ueye.c_int(Gain)
        if gain.value != current_gain:
            if gain.value < 0:
                gain = ueye.c_int(0)
                if i == 1:
                    print('Gain exceed lower limit >= 0')
            elif gain.value > 100:
                gain = ueye.c_int(100)
                if i == 1:
                    print('Gain exceed upper limit <= 100')
            gain_code = ueye.c_int(round(slope*gain.value+b))
            
            ueye.is_SetHWGainFactor(camPar.hCam, ueye.IS_SET_MASTER_GAIN_FACTOR, gain_code)
            new_gain = round((gain_code-b)/slope)
            
            if i == 1:
                print('               new GAIN = '+str(new_gain))
            gain_eff = new_gain
        ############################### Set GAIN Boost ################################
        ### Read current state of the gain boost
        current_gain_boost_bool = ueye.is_SetGainBoost(camPar.hCam, ueye.IS_GET_GAINBOOST)
        if nRet != ueye.IS_SUCCESS:
            print("Gain boost ERROR")   
        if current_gain_boost_bool == 0:
            current_gain_boost = 'OFF'
        elif current_gain_boost_bool == 1:
            current_gain_boost = 'ON'
        
        if i == 1:
            print('current Gain boost mode = ' + current_gain_boost)
    
        ### Set the state of the gain boost
        if gain_boost != current_gain_boost: 
            if gain_boost == 'OFF':
                nRet = ueye.is_SetGainBoost (camPar.hCam, ueye.IS_SET_GAINBOOST_OFF)                
                print('         new Gain Boost : OFF')
                
            elif gain_boost == 'ON':
                nRet = ueye.is_SetGainBoost (camPar.hCam, ueye.IS_SET_GAINBOOST_ON)                
                print('         new Gain Boost : ON')
                    
            if nRet != ueye.IS_SUCCESS:
                print("Gain boost setting ERROR")          
        ############################### Set Gamma #####################################
        ### Check boundary of Gamma
        if nGamma > 2.2:
            nGamma = 2.2
            if i == 1:
                print('Gamma exceed upper limit <= 2.2')
        elif nGamma < 1:
            nGamma = 1
            if i == 1:
                print('Gamma exceed lower limit >= 1')
        ### Read current Gamma    
        c_nGamma_init = ueye.c_void_p() 
        sizeOfnGamma = ueye.c_uint(4)    
        nRet = ueye.is_Gamma(camPar.hCam, ueye.IS_GAMMA_CMD_GET, c_nGamma_init, sizeOfnGamma)
        if nRet != ueye.IS_SUCCESS:
            print("Gamma getting ERROR") 
        else:
            if i == 1:
                print('          current Gamma = ' + str(c_nGamma_init.value/100))
        ### Set Gamma
            c_nGamma = ueye.c_void_p(round(nGamma*100)) # need to multiply by 100 [100 - 220]
            if c_nGamma_init.value != c_nGamma.value:
                nRet = ueye.is_Gamma(camPar.hCam, ueye.IS_GAMMA_CMD_SET, c_nGamma, sizeOfnGamma)
                if nRet != ueye.IS_SUCCESS:
                    print("Gamma setting ERROR") 
                else:
                    if i == 1:
                        print('              new Gamma = '+str(c_nGamma.value/100))
        ############################### Set Exposure time #############################
        ### Read current Exposure Time
        getExposure = ueye.c_double()
        sizeOfpParam = ueye.c_uint(8)   
        nRet = ueye.is_Exposure(camPar.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, getExposure, sizeOfpParam)
        if nRet == ueye.IS_SUCCESS:
            getExposure.value = round(getExposure.value*1000)/1000
            
            if i == 1:
                print('  current Exposure Time = ' + str(getExposure.value) + ' ms')
        ### Get minimum Exposure Time
        minExposure = ueye.c_double()
        nRet = ueye.is_Exposure(camPar.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, minExposure, sizeOfpParam)
        ### Get maximum Exposure Time
        maxExposure = ueye.c_double()
        nRet = ueye.is_Exposure(camPar.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, maxExposure, sizeOfpParam)
        ### Get increment Exposure Time
        incExposure = ueye.c_double()
        nRet = ueye.is_Exposure(camPar.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, incExposure, sizeOfpParam)
        ### Set new Exposure Time
        setExposure = ueye.c_double(ExposureTime)
        if setExposure.value > maxExposure.value:
           setExposure.value = maxExposure.value 
           if i == 1:
               print('Exposure Time exceed upper limit <= ' + str(maxExposure.value))
        elif setExposure.value < minExposure.value:
           setExposure.value = minExposure.value
           if i == 1:
               print('Exposure Time exceed lower limit >= ' + str(minExposure.value))
    
        if (setExposure.value < getExposure.value-incExposure.value/2) | (setExposure.value > getExposure.value+incExposure.value/2):   
            nRet = ueye.is_Exposure(camPar.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, setExposure, sizeOfpParam)
            if nRet != ueye.IS_SUCCESS:
                print("Exposure Time ERROR")
            else:
                if i == 1:
                    print('      new Exposure Time = ' + str(round(setExposure.value*1000)/1000) + ' ms')
        ############################### Set Black Level ###############################
        current_black_level_c = ueye.c_uint()      
        sizeOfBlack_level = ueye.c_uint(4)    
        ### Read current Black Level
        nRet = ueye.is_Blacklevel(camPar.hCam, ueye.IS_BLACKLEVEL_CMD_GET_OFFSET, current_black_level_c, sizeOfBlack_level)
        if nRet != ueye.IS_SUCCESS:
            print("Black Level getting ERROR")
        else:
            if i == 1:
                print('    current Black Level = ' + str(current_black_level_c.value))
            
        ### Set Black Level 
        if black_level > 255:
            black_level = 255
            if i == 1:
                print('Black Level exceed upper limit <= 255')
        if black_level < 0:
            black_level = 0
            if i == 1:
                print('Black Level exceed lower limit >= 0')
            
        black_level_c = ueye.c_uint(black_level)
        if black_level != current_black_level_c.value  :            
            nRet = ueye.is_Blacklevel(camPar.hCam, ueye.IS_BLACKLEVEL_CMD_SET_OFFSET, black_level_c, sizeOfBlack_level)
            if nRet != ueye.IS_SUCCESS:
                print("Black Level setting ERROR")
            else:
                if i == 1:
                    print('        new Black Level = ' + str(black_level_c.value))
            
    
    camPar.fps = round(dblFPS_eff.value*100)/100
    camPar.gain = gain_eff
    camPar.gainBoost = gain_boost
    camPar.gamma = c_nGamma.value/100
    camPar.exposureTime = round(setExposure.value*1000)/1000
    camPar.blackLevel = black_level_c.value
    
    return camPar
        
    
    


def snapshot(camPar, pathIDSsnapshot, pathIDSsnapshot_overview):
    """
    Snapshot of the IDS camera
    
    Args:
        CAM: a structure containing the parameters of the IDS camera
    """
    array = ueye.get_data(camPar.pcImageMemory, camPar.rectAOI.s32Width, camPar.rectAOI.s32Height, camPar.nBitsPerPixel, camPar.pitch, copy=False)
    
    # ...reshape it in an numpy array...
    frame = np.reshape(array,(camPar.rectAOI.s32Height.value, camPar.rectAOI.s32Width.value))#, camPar.bytes_per_pixel))
    
    with pathIDSsnapshot.open('wb') as f: #('ab') as f: #(pathname, mode='w', encoding='utf-8') as f: #('ab') as f:
        np.save(f,frame)
    
    maxi = np.amax(frame)
    if maxi == 0:
        maxi = 1
    im = Image.fromarray(frame*math.floor(255/maxi))
    im.save(pathIDSsnapshot_overview)
    
    maxi = np.amax(frame)
    # print()
    # print('frame max = ' + str(maxi))
    # print('frame min = ' + str(np.amin(frame)))
    if maxi >= 255:
        print('Saturation detected')
        
    plt.figure
    plt.imshow(frame)#, cmap='gray', vmin=mini, vmax=maxi)  
    plt.colorbar(); 
    plt.show()
        
    



def check_ueye(func, *args, exp=0, raise_exc=True, txt=None):
    ret = func(*args)
    if not txt:
        txt = "{}: Expected {} but ret={}!".format(str(func), exp, ret)
    if ret != exp:
        if raise_exc:
            raise RuntimeError(txt)
        else:
            logging.critical(txt)


def stopCapt_DeallocMem(camPar):
    # Stop capture and deallocate camera memory if need to change AOI
    if camPar.camActivated == 1:        
        nRet = ueye.is_StopLiveVideo(camPar.hCam, ueye.IS_FORCE_VIDEO_STOP)
        if nRet == ueye.IS_SUCCESS:
            camPar.camActivated = 0
            print('video stop successful')
        else:
            print('problem to stop the video')
            
    if camPar.Memory == 1:        
        nRet = ueye.is_FreeImageMem(camPar.hCam, camPar.pcImageMemory, camPar.MemID)
        if nRet == ueye.IS_SUCCESS:
            camPar.Memory = 0
            print('deallocate memory successful')
        else:
            print('Problem to deallocate memory of the camera')
                
    return camPar

def stopCapt_DeallocMem_ExitCam(camPar):
    # Stop capture and deallocate camera memory if need to change AOI
    if camPar.camActivated == 1:        
        nRet = ueye.is_StopLiveVideo(camPar.hCam, ueye.IS_FORCE_VIDEO_STOP)
        if nRet == ueye.IS_SUCCESS:
            camPar.camActivated = 0
            print('video stop successful')
        else:
            print('problem to stop the video')
            
    if camPar.Memory == 1:        
        nRet = ueye.is_FreeImageMem(camPar.hCam, camPar.pcImageMemory, camPar.MemID)
        if nRet == ueye.IS_SUCCESS:
            camPar.Memory = 0
            print('deallocate memory successful')
        else:
            print('Problem to deallocate memory of the camera')
    
    if camPar.Exit == 2:
        nRet = ueye.is_ExitCamera(camPar.hCam)
        if nRet == ueye.IS_SUCCESS:
            camPar.Exit = 0
            print('Camera disconnected')
        else:
            print('Problem to disconnect camera, need to restart spyder')
                
    return camPar

class ImageBuffer:
    pcImageMemory = None
    MemID = None
    width = None
    height = None
    nbitsPerPixel = None

def imageQueue(camPar):
    #   Create Imagequeue ---------------------------------------------------------
    # - allocate 3 ore more buffers depending on the framerate
    # - initialize Imagequeue
    # ---------------------------------------------------------
    sleep(1)   # is required (delay of 1s was not optimized!!)
    buffers = []
    for y in range(10):
        buffers.append(ImageBuffer())

    for x in range(len(buffers)):
        buffers[x].nbitsPerPixel = camPar.nBitsPerPixel  # RAW8
        buffers[x].height = camPar.rectAOI.s32Height  # sensorinfo.nMaxHeight
        buffers[x].width = camPar.rectAOI.s32Width  # sensorinfo.nMaxWidth
        buffers[x].MemID = ueye.int(0)
        buffers[x].pcImageMemory = ueye.c_mem_p()
        check_ueye(ueye.is_AllocImageMem, camPar.hCam, buffers[x].width, buffers[x].height, buffers[x].nbitsPerPixel,
                   buffers[x].pcImageMemory, buffers[x].MemID)
        check_ueye(ueye.is_AddToSequence, camPar.hCam, buffers[x].pcImageMemory, buffers[x].MemID)

    check_ueye(ueye.is_InitImageQueue, camPar.hCam, ueye.c_int(0))
    if camPar.trigger_mode == 'soft':
        check_ueye(ueye.is_SetExternalTrigger, camPar.hCam, ueye.IS_SET_TRIGGER_SOFTWARE)
    elif camPar.trigger_mode == 'hard':
        check_ueye(ueye.is_SetExternalTrigger, camPar.hCam, ueye.IS_SET_TRIGGER_LO_HI)

def prepareCam(camPar, metadata):
    cam_path = metadata.output_directory + '\\' + metadata.experiment_name + '_video.' + camPar.vidFormat
    strFileName = ueye.c_char_p(cam_path.encode('utf-8'))
    
    if camPar.vidFormat == 'avi':         
        # print('Video format : AVI')
        camPar.avi = ueye.int()
        nRet = ueye_tools.isavi_InitAVI(camPar.avi, camPar.hCam)
        # print("isavi_InitAVI")
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("isavi_InitAVI ERROR")
        
        nRet = ueye_tools.isavi_SetImageSize(camPar.avi, camPar.m_nColorMode,  camPar.rectAOI.s32Width , camPar.rectAOI.s32Height, 0, 0, 0)
        nRet = ueye_tools.isavi_SetImageQuality(camPar.avi, 100)
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("isavi_SetImageQuality ERROR")
 
        nRet = ueye_tools.isavi_OpenAVI(camPar.avi, strFileName)
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("isavi_OpenAVI ERROR")
            print('Error code = ' + str(nRet))
            print('Certainly, it is a problem with the file name, Avoid special character like "µ" or try to redcue its size')
        
        nRet = ueye_tools.isavi_SetFrameRate(camPar.avi, camPar.fps)
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("isavi_SetFrameRate ERROR")
        nRet = ueye_tools.isavi_StartAVI(camPar.avi)
        # print("isavi_StartAVI")
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("isavi_StartAVI ERROR")

            
    elif camPar.vidFormat == 'bin':
        camPar.punFileID = ueye.c_uint()
        nRet = ueye_tools.israw_InitFile(camPar.punFileID, ueye_tools.IS_FILE_ACCESS_MODE_WRITE)
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("INIT RAW FILE ERROR")
            
        nRet = ueye_tools.israw_SetImageInfo(camPar.punFileID, camPar.rectAOI.s32Width, camPar.rectAOI.s32Height, camPar.nBitsPerPixel)
        if nRet != ueye_tools.IS_AVI_NO_ERR:
            print("SET IMAGE INFO ERROR")
            
        if nRet == ueye.IS_SUCCESS: 
            # print('initFile ok')
            # print('SetImageInfo ok')
            nRet = ueye_tools.israw_OpenFile(camPar.punFileID, strFileName)
            # if nRet == ueye.IS_SUCCESS:
            #     # print('OpenFile success')
    
    # nShutterMode = ueye.c_uint(ueye.IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING_GLOBAL_START)
    # nRet = ueye.is_DeviceFeature(camPar.hCam, ueye.IS_DEVICE_FEATURE_CMD_SET_SHUTTER_MODE, nShutterMode, 
    #                         ueye.sizeof(nShutterMode))
    # print('shutter mode = ' + str(nShutterMode.value) + ' / enable : ' + str(nRet))
    
    # # Read the global flash params
    # flashParams = ueye.IO_FLASH_PARAMS()
    # nRet = ueye.is_IO(camPar.hCam, ueye.IS_IO_CMD_FLASH_GET_GLOBAL_PARAMS, flashParams, ueye.sizeof(flashParams))
    # if (nRet == ueye.IS_SUCCESS):
    #     nDelay   = flashParams.s32Delay
    #     print('nDelay = ' + str(nDelay.value))
    #     nDuration = flashParams.u32Duration
    #     print('nDuration = ' + str(nDuration.value))

    # flashParams.s32Delay.value = 0
    # flashParams.u32Duration.value = 40 
    # # Apply the global flash params and set the flash params to these values
    # nRet = ueye.is_IO(camPar.hCam, ueye.IS_IO_CMD_FLASH_SET_PARAMS, flashParams, ueye.sizeof(flashParams))
    
    
    # nRet = ueye.is_IO(camPar.hCam, ueye.IS_IO_CMD_FLASH_GET_PARAMS, flashParams, ueye.sizeof(flashParams))
    # if (nRet == ueye.IS_SUCCESS):
    #     nDelay   = flashParams.s32Delay
    #     print('nDelay = ' + str(nDelay.value))
    #     nDuration = flashParams.u32Duration
    #     print('nDuration = ' + str(nDuration.value))
            
    # ---------------------------------------------------------
    # Activates the camera's live video mode (free run mode)
    # ---------------------------------------------------------
    nRet = ueye.is_CaptureVideo(camPar.hCam, ueye.IS_DONT_WAIT)
    # nRet = ueye.is_FreezeVideo(camPar.hCam, ueye.IS_DONT_WAIT)
    if nRet != ueye.IS_SUCCESS:
        print("is_CaptureVideo ERROR")
    else:
        camPar.camActivated = 1
    
    return camPar
    
        
def runCam_thread(camPar, start_chrono): 
    imageinfo = ueye.UEYEIMAGEINFO()
    current_buffer = ueye.c_mem_p()
    current_id = ueye.int()
    # inc = 0
    entier_old = 0
    # time.sleep(0.01)
    while True: 
        nret = ueye.is_WaitForNextImage(camPar.hCam, camPar.timeout, current_buffer, current_id)
        if nret == ueye.IS_SUCCESS:
            check_ueye(ueye.is_GetImageInfo, camPar.hCam, current_id, imageinfo, ueye.sizeof(imageinfo))
            start_time = time.time()
            counter = start_time - start_chrono
            camPar.time_array.append(counter)
            if camPar.vidFormat == 'avi':
                nRet = ueye_tools.isavi_AddFrame(camPar.avi, current_buffer)  
            elif camPar.vidFormat == 'bin':   
                nRet = ueye_tools.israw_AddFrame(camPar.punFileID, current_buffer, imageinfo.u64TimestampDevice)
                                
            check_ueye(ueye.is_UnlockSeqBuf, camPar.hCam, current_id, current_buffer)
        else:
            # nRet = ueye.is_FreeImageMem (camPar.hCam, current_buffer, current_id)
            # if nRet != ueye.IS_SUCCESS:
            #     print('ERROR to free the memory')
            #     print(nRet)
            print('Thread finished')
            break
        # else:
        #     print('thread cam stop correctly')
        #     break

def stopCam(camPar):
    if camPar.vidFormat == 'avi':
        ueye_tools.isavi_StopAVI(camPar.hCam)
        ueye_tools.isavi_CloseAVI(camPar.hCam)
        ueye_tools.isavi_ExitAVI(camPar.hCam)
    elif camPar.vidFormat == 'bin':   
        ueye_tools.israw_CloseFile(camPar.punFileID)
        ueye_tools.israw_ExitFile(camPar.punFileID)
        camPar.punFileID = ueye.c_uint()
        
    # camPar = stopCapt_DeallocMem(camPar)
    
    return camPar
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
