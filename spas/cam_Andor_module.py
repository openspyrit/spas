# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:48:30 2025

@author: mahieu

Control of the Andor Zyla cameras with the official Andor package
"pyAndorSDK3" (Andor SDK3).
The previous version, based on pylablib, is in cam_Andor_module_pylablib_pack.py

Install (in the conda env, from a writable copy of the folder):
    pip install "C:/Program Files/Andor SDK3/Python/pyAndorSDK3"

Warning: Solis must be closed, otherwise the cameras are already in use.

Example:
    cam_spat = AndorCam()
    cam_spat.init(model = 'ZYLA-4.2P-USB3-S', SN = 'VSC-10323', arm = 'spatial')
    cam_spat_params = cam_spat.setup(expos_time = 0.05, binningX = 1, binningY = 1, snapshot = True)
    data = cam_spat.snapshot(tilt_image = True)
    cam_spat.display(display_max = True)
    cam_spat.disconnect()

To read a feature : cam_spat.get("ExposureTime"), to write it : cam_spat.set("ExposureTime", 0.1)
The list of the features is in the SDK3 manual (C:/Program Files/Andor SDK3/Docs)
"""

from pyAndorSDK3 import AndorSDK3, CameraException, ATCoreException
from pyAndorSDK3.andor_utility import ATUtility
import numpy as np
from matplotlib import pyplot as plt
from typing import Optional
from dataclasses import dataclass, InitVar
from dataclasses_json import dataclass_json
from scipy import signal
from collections import deque
import cv2


# The SDK3 library is initialized once and shared by the cameras:
# it is finalized when the AndorSDK3 object is deleted.
_sdk3 = None
# indexes of the cameras already opened by an AndorCam object
_opened_indexes = set()


def get_sdk3() -> AndorSDK3:
    """Return the AndorSDK3 object shared by all the cameras (created at the first call)."""
    global _sdk3
    if _sdk3 is None:
        _sdk3 = AndorSDK3()

    return _sdk3


class AndorCam:
    """Class that controls an Andor camera (Zyla) with the SDK3.

    Attributes:
        sdk (pyAndorSDK3 Camera):
            the Andor object of the camera. The features can be read and written directly,
            e.g. cam.sdk.ExposureTime. Can be used for functions not implemented here.
        index (int):
            the index of the camera in the SDK3.
        model (str):
            the model of the camera read from the camera (feature "CameraModel").
        serial_number (str):
            the serial number of the camera.
        arm (str):
            'spatial' or 'spectral', the arm of the SPIM where the camera is.
        snapshot_mode (bool):
            if False => acquire video, if True => acquire an image.
        nframes (int):
            the number of buffers (frames) queued for the acquisition.
        last_timestamp (float):
            the timestamp (s) of the last frame read, from the clock of the camera
            (start of the exposure, read in the metadata of the frame).
    """

    def __init__(self):
        self.sdk = None
        self.index = None
        self.model = None
        self.serial_number = None
        self.arm = None
        self.snapshot_mode = False
        self.nframes = 2
        self.last_timestamp = None
        # buffers queued in the SDK, in the order they will be filled
        self._buffers = deque()
        self._image_size = None
        self._aoi = None
        self._metadata_timestamp = False
        self._clock_frequency = None


    def init(self, model: str = 'Zyla', SN: str = '', arm: str = 'spatial'):
        """
        Initialize the camera

        Parameters
        ----------
        model : str
            the model of the camera, compared (case insensitive) to the "CameraModel" feature
            read from the camera, e.g. 'Zyla' matches 'ZYLA-4.2P-USB3'. The default is 'Zyla'.
        SN : str
            enter the serial number of the camera.
        arm : str
            'spatial' or 'spectral', the arm of the SPIM where the camera is.
        """
        sdk3 = get_sdk3()
        cameras_found = []
        for index in range(sdk3.DeviceCount):
            if index in _opened_indexes:
                continue
            try:
                camera = sdk3.GetCamera(index)
            except CameraException:
                continue
            serial_number = camera.SerialNumber
            if serial_number == SN:
                self.sdk = camera
                self.index = index
                break
            cameras_found.append(camera.CameraModel + ' SN = ' + serial_number)
            camera.close()

        if self.sdk is None:
            raise RuntimeError('camera ANDOR, SN = ' + SN + ' not found. Free cameras detected : ' + str(cameras_found) +
                               '. Check that the camera is switched on and that Solis (or another Python kernel) does not use it.')

        camera_model = self.sdk.CameraModel
        if model.lower() not in camera_model.lower():
            self.sdk.close()
            self.sdk = None
            raise RuntimeError('camera ANDOR, SN = ' + SN + ' is a ' + camera_model + ', not a ' + model)

        _opened_indexes.add(self.index)
        self.model = camera_model
        self.serial_number = SN
        self.arm = arm
        print('camera ANDOR ' + camera_model + ', SN = ' + SN + ' connected and dedicated to the ' + arm.upper() + ' arm')


    def disconnect(self):
        """disconnect the camera"""
        self.stop_acquisition()
        self.sdk.close()
        _opened_indexes.discard(self.index)
        self.sdk = None
        self.index = None
        print(self.arm + ' camera disconnected')


    def get(self, feature: str):
        """Read the value of a feature of the camera (e.g. "ExposureTime")."""
        return getattr(self.sdk, feature)


    def set(self, feature: str, value):
        """Write the value of a feature of the camera (e.g. "ExposureTime")."""
        setattr(self.sdk, feature, value)


    def bit_depth(self) -> int:
        """Return the bit depth of the image (e.g. 12 or 16)."""
        image_bit_depth_str = self.get("BitDepth")

        return int(image_bit_depth_str[:image_bit_depth_str.index(' Bit')])

    ########################### acquisition ###################################
    def is_acquiring(self) -> bool:
        """Return True if an acquisition is in progress."""
        return bool(self.get("CameraAcquiring"))


    def setup_acquisition(self, nframes: int = 2):
        """Set the number of buffers (frames) queued for the next acquisition.
        It must be large enough to store the frames that arrive before being read."""
        self.nframes = max(int(nframes), 1)


    def min_trigger_period(self) -> float:
        """Return the minimum time (s) between two external triggers so that no trigger is ignored.

        Without Overlap, a trigger received while the previous frame is read out is ignored by the camera:
        the period must be longer than the exposure time + the readout time.
        With Overlap, the exposure of a frame is done during the readout of the previous one.
        """
        exposure_time = self.get("ExposureTime")
        readout_time = self.get("ReadoutTime")
        if self.get("Overlap"):
            return max(exposure_time, readout_time)

        return exposure_time + readout_time


    def start_acquisition(self):
        """Queue the buffers and start a continuous acquisition. Nothing to do if already started.

        The buffers are queued directly in the SDK (and not with the Camera functions of pyAndorSDK3,
        which need the "MetadataFrameInfo" feature that the Zyla does not have when the metadata are enabled).
        """
        if self.is_acquiring():
            return

        self.set("CycleMode", "Continuous")
        self._flush()
        self._image_size = self.get("ImageSizeBytes")
        self._aoi = (self.get("AOIHeight"), self.get("AOIWidth"), self.get("AOIStride"), self.get("PixelEncoding"))
        self._metadata_timestamp = bool(self.get("MetadataEnable")) and bool(self.get("MetadataTimestamp"))
        self._clock_frequency = self.get("TimestampClockFrequency")
        for _ in range(self.nframes):
            self._queue(np.empty(self._image_size, dtype=np.uint8))
        self.sdk.AcquisitionStart()


    def stop_acquisition(self):
        """Stop the acquisition and release the buffers."""
        if self.is_acquiring():
            self.sdk.AcquisitionStop()
        self._flush()


    def _queue(self, buffer: np.ndarray):
        """Give a buffer to the SDK, it will be filled by a next frame."""
        self.sdk.lib.queue_buffer(self.sdk.handle, buffer.ctypes.data, self._image_size)
        self._buffers.append(buffer)


    def _flush(self):
        """Release all the buffers queued in the SDK."""
        self.sdk.lib.flush(self.sdk.handle)
        self._buffers.clear()


    def _decode(self, buffer: np.ndarray) -> np.ndarray:
        """Convert the raw buffer into a 2D image (height, width)."""
        (height, width, stride, encoding) = self._aoi
        data = buffer[:height * stride]
        if encoding == "Mono12Packed":
            image = np.empty(height * width, dtype=np.uint16)
            ATUtility().unpack(data.ctypes.data, image.ctypes.data, width, height, stride, "Mono12Packed", "Mono16")
            return image.reshape(height, width)
        elif encoding in ("Mono12", "Mono16"):
            return data.view(np.uint16).reshape(height, stride // 2)[:, :width].copy()
        elif encoding == "Mono32":
            return data.view(np.uint32).reshape(height, stride // 4)[:, :width].copy()
        else:
            raise ValueError('Pixel encoding ' + encoding + ' not supported')


    def read_frame(self, timeout: float = 5) -> np.ndarray:
        """Wait for the next frame of the acquisition and return it.
        Its timestamp (s) is stored in self.last_timestamp.

        Parameters
        ----------
        timeout : float
            the maximum time to wait the frame (s).

        Returns
        -------
        image : np.ndarray
            2D array (height, width) of uint16.
        """
        try:
            (buffer_ptr, _) = self.sdk.lib.wait_buffer(self.sdk.handle, timeout * 1000)
        except ATCoreException as e:
            raise RuntimeError(self.arm + ' camera: no frame received after ' + str(timeout) + ' s (SDK3 error ' + str(e) + ')')

        buffer = self._buffers.popleft()
        if int(self.sdk.lib.ffi.cast("uintptr_t", buffer_ptr[0])) != buffer.ctypes.data:
            raise RuntimeError(self.arm + ' camera: the frame returned by the SDK is not in the expected buffer')

        image = self._decode(buffer)
        if self._metadata_timestamp:
            ticks = ATUtility().getTimeStampFromMetadata(buffer.ctypes.data, self._image_size)
        else:
            ticks = self.get("TimestampClock")
        self.last_timestamp = ticks / self._clock_frequency
        # the buffer is re-queued to be used for the next frames
        self._queue(buffer)

        return image


    def snap(self, timeout: Optional[float] = None) -> np.ndarray:
        """Acquire one frame. If an acquisition is in progress, the next frame is read."""
        if timeout is None:
            timeout = self.get("ExposureTime") + 5

        if self.is_acquiring():
            return self.read_frame(timeout)

        nframes = self.nframes
        self.setup_acquisition(1)
        self.start_acquisition()
        try:
            image = self.read_frame(timeout)
        finally:
            self.stop_acquisition()
            self.setup_acquisition(nframes)

        return image

    ########################### setup #########################################
    def setup(self, expos_time: float = 0.1, ExternalTriggerDelay: float = 0, gain: int = 1,
              baseline: int = 100, width: int = 2048, height: int = 2048, offsetX: int = 1, offsetY: int = 1,
              binningX: int = 1, binningY: int = 1, encodPix: int = 12, snapshot: bool = False):
        """
        setup the Andor camera

        Parameters
        ----------
        expos_time : float, optional
            the exposure time in s. The default is 0.1.
        ExternalTriggerDelay. float.
        the delay between the trigger received and to start the acquisition (s)
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
        cam_Parameters (obj)
            the metadata containing the camera parameters.

        """
        self.stop_acquisition()
        ########################### setting binning ###############################
        binX = self.get("AOIHBin")
        if binX != binningX:
            self.set("AOIHBin", binningX)
            print('binning X set to: ' + str(binningX))

        binY = self.get("AOIVBin")
        if binY != binningY:
            self.set("AOIVBin", binningY)
            print('binning Y set to: ' + str(binningY))
        ########################### setting ROI ###################################
        width_max = int(2048/binningX)
        height_max = int(2048/binningY)

        offsetX_cur = self.get("AOILeft")
        offsetY_cur = self.get("AOITop")

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

        if offsetX_cur > 0 and offsetX_acc == 0:
            self.set("AOILeft", offsetX_acc)
            self.set("AOIWidth", width_acc)
        else:
            self.set("AOIWidth", width_acc)
            self.set("AOILeft", offsetX_acc)

        if offsetY_cur > 0 and offsetY_acc == 0:
            self.set("AOITop", offsetY_acc)
            self.set("AOIHeight", height_acc)
        else:
            self.set("AOIHeight", height_acc)
            self.set("AOITop", offsetY_acc)

        print('width set to    : ' + str(self.get("AOIWidth")))
        print('height set to   : ' + str(self.get("AOIHeight")))
        print('offset X set to : ' + str(self.get("AOILeft")))
        print('offset Y set to : ' + str(self.get("AOITop")))
        ############### set the Spurious Noise Filter ############################# to reduce the "salt & pepper noise rendering
        SpuriousNoiseFilter = bool(self.get("SpuriousNoiseFilter"))
        print("Spurious Noise Filter is", SpuriousNoiseFilter)
        if SpuriousNoiseFilter == False:
            self.set("SpuriousNoiseFilter", True)
            print("SpuriousNoiseFilter set to", bool(self.get("SpuriousNoiseFilter")))
        ############### set the Blemish Correction Filter ######################## to take off hot spike/unresponsive pixels
        StaticBlemishCorrection = bool(self.get("StaticBlemishCorrection"))
        print("Static Blemish Correction is", StaticBlemishCorrection)
        if StaticBlemishCorrection == False:
            self.set("StaticBlemishCorrection", True)
            print("StaticBlemishCorrection set to", bool(self.get("StaticBlemishCorrection")))
        ################## read Pixel Readout Rate #################################
        self.set("PixelReadoutRate", '100 MHz')# other possibiliti is "270 MHz", it is faster and noiser
        PixelReadoutRate = self.get("PixelReadoutRate")
        print("Pixel Readout Rate =", PixelReadoutRate)
        #################### set Trigger Mode #####################################
        # With the spectral Zyla (ZYLA-4.2P-USB3), the minimum exposure time in External mode does not decrease
        # below the exposure time used in Internal mode before (e.g. 984 µs instead of 24 µs): the exposure time
        # is set to its minimum in Internal mode before switching to External, to have the real minimum.
        self.set("TriggerMode", "Internal")
        self.set("ExposureTime", self.get("min_ExposureTime"))
        self.set("TriggerMode", "External")
        print("Trigger Mode is :", self.get("TriggerMode"))
        ################# set External Trigger Delay ##############################
        ExternalTriggerDelay_current = self.get("ExternalTriggerDelay")
        if ExternalTriggerDelay_current != ExternalTriggerDelay:
            self.set("ExternalTriggerDelay", ExternalTriggerDelay)
            print("External Trigger Delay =", self.get("ExternalTriggerDelay"), "s")
        else:
            print("External Trigger Delay already set to =", ExternalTriggerDelay_current)
        ############### read the SimplePreAmpGainControl ##########################
        if encodPix == 16:
            SimplePreAmpGainControl = "16-bit (low noise & high well capacity)"
            encodingPixel = "Mono16"
        else:
            if encodPix != 12:
                print("Warning, encoding pixel has a bad entry value. default is taken (12 bits)")
            SimplePreAmpGainControl = "12-bit (low noise)"
            encodingPixel = "Mono12Packed"

        self.set("SimplePreAmpGainControl", SimplePreAmpGainControl)
        print("Simple PreAmp Gain Control =", self.get("SimplePreAmpGainControl"))
        ################# read the Bit depth ######################################
        print("BitDepth =", self.get("BitDepth"))
        ##################### Encoding Pixel ######################################
        self.set("PixelEncoding", encodingPixel)
        print("Pixel Encoding =", self.get("PixelEncoding"))
        ##################### timestamp in the metadata ###########################
        # the timestamp of the start of the exposure is added at the end of each frame
        self.set("MetadataEnable", True)
        self.set("MetadataTimestamp", True)
        #################### setting the exposure time ###########################
        # the limits are read from the camera, they depend on the readout rate, the AOI, the binning and the trigger mode
        exposure_mini = self.get("min_ExposureTime")
        exposure_maxi = self.get("max_ExposureTime")
        print('exposure time limits : ' + str(exposure_mini) + ' s to ' + str(exposure_maxi) + ' s')
        exposure_time = expos_time
        if exposure_time < exposure_mini:
            exposure_time = exposure_mini
            print('the exposure time is below the minimum value, it is set to ' + str(exposure_time))
        if exposure_time > exposure_maxi:
            exposure_time = exposure_maxi
            print('the exposure time is above the maximum value, it is set to ' + str(exposure_time))

        self.set("ExposureTime", exposure_time)
        print('exposure time set to : ' + str(self.get("ExposureTime")) + ' s')
        ######################### get the frame rate ##############################
        print('frame rate = ' + str(self.get("FrameRate")) + ' fps')
        print('readout time = ' + str(round(self.get("ReadoutTime") * 1e3, 3)) + ' ms, minimum period between two triggers = ' +
              str(round(self.min_trigger_period() * 1e3, 3)) + ' ms')
        ########################## setting gain ###################################
        curent_gain = self.get("PreAmpGain")
        if curent_gain != 'x' + str(gain):
            self.set("PreAmpGain", 'x' + str(gain))
            print('the gain is set to : ' + str(self.get("PreAmpGain")))
        ################ acquisition mode: snapshot or video #####################
        self.snapshot_mode = snapshot

        return cam_Parameters(cam = self)

    ########################### display #######################################
    def snapshot(self, tilt_image: bool = False):
        """
        take a snapshot of the camera, display it and print the SNR

        Parameters:
        -----------
             tilt_image (bool):
                 Tilte the image as display on the DMD. Default is False.
        Returns
        -------
        data (np.ndarray):
            the image.

        """
        image_bit_depth = self.bit_depth()

        data = self.snap()
        ########################• 2d median filter ################################
        data = signal.medfilt2d(data, kernel_size=3)
        ######################## check saturation #################################
        if np.max(data) >= 2**image_bit_depth - 1:
            print('!!!!!!!!!! Warning, saturation detected !!!!!!!!!!!!')
        ########################## tilt image ####################################
        if tilt_image:
            data = np.rot90(data, k=1, axes=(0,1))
            print('image tilted')
        ########################## print snapshot #################################
        plt.figure()
        plt.imshow(data)
        plt.colorbar()
        plt.title(self.arm + ' cam')
        plt.xlabel('X (Width)')
        plt.ylabel('Y (Height)')

        Sig = np.max(data[round(data.shape[0]/2 - 100):round(data.shape[0]/2 + 100), round(data.shape[1]/2 - 100):round(data.shape[1]/2 + 100)])
        noise = np.std(data[round(50/(data.shape[0]/self.get("AOIHeight"))):round(150/(data.shape[0]/self.get("AOIHeight"))), round(data.shape[1]/2 - 100):round(data.shape[1]/2 + 100)].flatten())
        maxi = np.max(data)
        print('Original :')
        print('     max   = ' + str(maxi))
        print('     Sig   = ' + str(Sig))
        print('     Noise = ' + str(noise))
        print('     SNR   = ' + str(Sig/noise))

        return data


    def display(self, display_max: bool = False,
                display_integral: bool = False, display_profile: bool = False):
        """
        Continuous image display of the camera with optional integral/mean curve using OpenCV.
        Press 'q' to exit.

        Parameters:
        -----------
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
        width = self.get("AOIWidth")
        height = self.get("AOIHeight")
        if self.arm == 'spatial':
            # the image is rotated
            width, height = height, width
        ratio = width / height
        height_win = 900
        width_win = int(height_win * ratio)

        image_bit_depth = self.bit_depth()
        nframes = self.nframes

        try:
            # Create OpenCV window for the camera
            if self.arm == 'spatial':
                window_name = "Camera of the Spatial Arm"
            else:
                window_name = "Camera of the Spectral Arm"

            cv2.namedWindow(window_name)

            # Trackbar callback (does nothing)
            def nothing(x):
                pass

            # Exposure time setup
            min_exposure_time = self.get("min_ExposureTime") * 1e6
            max_exposure_time = self.get("max_ExposureTime") * 1e6
            current_exposure_time = self.get("ExposureTime")

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
                curve_height = 200  # Hauteur de l'image de la courbe
                fixed_width = 800    # Largeur fixe de la fenêtre
                # Initialiser l'image pour la courbe (noire, 1 canal)
                curve_img = np.zeros((curve_height, fixed_width), dtype=np.uint8)

            # Start data acquisition, few buffers to display the last frames
            print('Start acquisition...\n')
            self.setup_acquisition(3)
            self.start_acquisition()

            while True:
                # Acquire image
                data = self.read_frame(timeout=current_exposure_time + 5)

                if self.arm == 'spatial':
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
                    print('Maximum exposure time set to ' + str(max_exposure_time / 1e6) + ' s')
                    exposure_time = max_exposure_time

                if abs(exposure_time / 1e6 - current_exposure_time) > 1e-6:
                    self.set("ExposureTime", exposure_time / 1e6)
                    current_exposure_time = self.get("ExposureTime")

                # Process image
                data_64b = data_8b.astype(np.float64)
                data2 = data_64b * brightness / max(maxi, 1)
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

                if display_profile:
                    # define offset and thickness of the profile
                    offset = int(data_64b.shape[0]/2)
                    half_thick = 50
                    # Extraire le profil (colonne centrale)
                    data_rogne = data[offset - half_thick:offset + half_thick, :]

                    profile = np.mean(data_rogne, axis=0)
                    profile = profile[10: len(profile) - 10]
                    # Normaliser le profil pour l'affichage
                    y_min, y_max = np.min(profile), np.max(profile)
                    y_range = y_max - y_min if y_max != y_min else 1

                    # Effacer l'image de la courbe
                    curve_img.fill(0)

                    # Échantillonner le profil pour qu'il tienne dans fixed_width
                    step = len(profile) / fixed_width
                    points = []
                    for i in range(fixed_width):
                        idx = int(i * step)
                        if idx >= len(profile):
                            break
                        y = int(curve_height - (profile[idx] - y_min) / y_range * curve_height)
                        points.append((i, y))

                    # Tracer le profil
                    if len(points) > 1:
                        cv2.polylines(curve_img, [np.array(points, dtype=np.int32)], False, 255, 1)

                    # Tracer les axes
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
            self.stop_acquisition()
            self.setup_acquisition(nframes)
            print('Stopping acquisition...')


@dataclass_json
@dataclass
class cam_Parameters:
    """
    Class containing the camera parameters, read from the camera.
    """
    arm: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
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

    simplePreAmpGainControl: Optional[str] = None
    encodPix: Optional[int] = None
    snapshot: Optional[bool] = None

    cam: InitVar[AndorCam] = None

    class_description: str = None

    def __post_init__(self, cam: Optional[AndorCam] = None):
        if cam is None:
            return

        self.arm = cam.arm
        self.model = cam.model
        self.serial_number = cam.serial_number
        self.exposure_time_s = cam.get("ExposureTime")
        self.exposure_time_µs = self.exposure_time_s * 1e6
        self.frame_rate = cam.get("FrameRate")
        self.gain = cam.get("PreAmpGain")

        self.width  = cam.get("AOIWidth")
        self.height = cam.get("AOIHeight")
        self.offsetX  = cam.get("AOILeft")
        self.offsetY = cam.get("AOITop")
        self.binningX = cam.get("AOIHBin")
        self.binningY = cam.get("AOIVBin")

        self.bit_depth = cam.get("BitDepth")
        self.BytesPerPixel = cam.get("BytesPerPixel")

        self.Baseline = cam.get("Baseline")
        self.ExternalTriggerDelay = cam.get("ExternalTriggerDelay")
        self.FanSpeed = cam.get("FanSpeed")
        self.FastAOIFrameRateEnable = bool(cam.get("FastAOIFrameRateEnable"))
        self.MetadataEnable = bool(cam.get("MetadataEnable"))
        self.MetadataTimestamp = bool(cam.get("MetadataTimestamp"))
        self.Overlap = bool(cam.get("Overlap"))
        self.PreAmpGainSelector = cam.get("PreAmpGainSelector")
        self.SimplePreAmpGainControl = cam.get("SimplePreAmpGainControl")
        self.SpuriousNoiseFilter = bool(cam.get("SpuriousNoiseFilter"))
        self.StaticBlemishCorrection = bool(cam.get("StaticBlemishCorrection"))
        self.TriggerMode = cam.get("TriggerMode")
        self.simplePreAmpGainControl = self.SimplePreAmpGainControl

        encodPixel = cam.get("PixelEncoding")
        if encodPixel.find("Mono12Packed") == 0:
            encodPixel = 12
        elif encodPixel.find("Mono16") == 0:
            encodPixel = 16
        else:
            print("warning, problem to detect the enconding Pixel, default is 12 bit")
            encodPixel = 12

        self.encodPix = encodPixel
        self.snapshot = cam.snapshot_mode
        self.class_description = cam.arm + ' camera parameters'
