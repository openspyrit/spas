# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 2026

@author: mahieu

Translation stage of the collection objective : Thorlabs actuator Z825B (DC servo motor, travel 25 mm)
driven by the T-Cube DC servo controller TDC001 (Kinesis).
"""

import clr
import os
import sys
import time
import numpy as np
from System import Decimal

sys.path.append(r"C:/Program Files/Thorlabs/Kinesis")

clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.DCServoCLI")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI, DeviceConfiguration
from Thorlabs.MotionControl.TCube.DCServoCLI import TCubeDCServo

TIMEOUT_MOVE_MS = 60000     # Z825B : 25 mm at 2.3 mm/s ~ 11 s, homing is slower
TRAVEL_MM = (0.0, 25.0)     # travel range of the Z825B
# position saved at the disconnection, to come back to it at the next session
LAST_POSITION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stage_collect_obj_last_position.txt')
# autofocus : coarse scan of +/- AF_HALF_RANGE_MM around the current position with a step AF_STEP_MM,
# then fine scan of +/- AF_STEP_MM around the best position with a step AF_STEP_MM / 5
AF_HALF_RANGE_MM = 0.5
AF_STEP_MM = 0.05
AF_MAX_EXTENSIONS = 2      # new coarse scans if the best sharpness is at the edge of the previous one
AF_BINNING = 4             # binning of the image before the computation of the sharpness (noise, speed)
BACKLASH_MM = 0.05          # the positions are always reached from below (same direction as the scans)


class AutofocusStopped(Exception):
    """The autofocus was stopped by the user."""


def sharpness(image: np.ndarray, binning: int = AF_BINNING) -> float:
    """Sharpness of an image : energy of its gradient (Tenengrad) divided by its squared mean intensity,
    so that it does not depend on the illumination (laser power, exposure time)."""
    image = np.asarray(image, dtype=np.float64)
    height, width = (image.shape[0] // binning) * binning, (image.shape[1] // binning) * binning
    image = image[:height, :width].reshape(height // binning, binning, width // binning, binning).mean(axis=(1, 3))
    grad_y, grad_x = np.gradient(image)
    return float(np.mean(grad_x**2 + grad_y**2) / max(image.mean(), 1e-9)**2)


class CollectObjStage:
    """Class that controls the translation stage of the collection objective : actuator Thorlabs Z825B
    driven by the T-Cube DC servo controller TDC001 (Kinesis).

    Example:
        obj_stage = CollectObjStage()
        obj_stage.init(model = 'Z825B', SN = '83850559')
        obj_stage.home()
        obj_stage.go_to_last_position(verbose = True)
        obj_stage.move_to(12.5)
        obj_stage.move_by(-0.1)
        position = obj_stage.read_position(verbose = True)
        obj_stage.autofocus(cam_spat)   # or the AUTOFOCUS button of stage.stage_adjustment(obj_stage = obj_stage, cam = cam_spat)
        obj_stage.disconnect(go_home = True)
    """

    travel = TRAVEL_MM   # travel range (mm), used by the manual control window of PIStage.stage_adjustment()
    # defaults for an object created before a reload of the module (Spyder)
    af_half_range = AF_HALF_RANGE_MM
    af_step = AF_STEP_MM

    def __init__(self):

        self.device = None
        self.model = None
        self.serial_number = None
        self.af_half_range = AF_HALF_RANGE_MM   # (mm) settings of autofocus(), can be changed in the console
        self.af_step = AF_STEP_MM


    def init(self, model: str = 'Z825B', SN: str = '', home: bool = False):
        """
        Initialize the stage

        Parameters
        ----------
        model : str
            the actuator connected to the controller, used to load its settings (pitch, velocity, travel)
            from the Kinesis database : the TDC001 cannot detect the actuator. The default is 'Z825B'.
        SN : str
            the serial number of the TDC001 controller.
        home : bool
            if True, the stage is homed (moved to its 0 mm limit switch) after the connection. The absolute
            positions are meaningful only after a homing. The default is False.
        """
        DeviceManagerCLI.BuildDeviceList()
        try:
            self.device = TCubeDCServo.CreateTCubeDCServo(SN)
            self.device.Connect(SN)
        except Exception as e:
            detected = list(DeviceManagerCLI.GetDeviceList(TCubeDCServo.DevicePrefix))
            self.device = None
            raise RuntimeError('stage of the collection objective, SN = ' + SN + ' not connected (' + str(e) + '). DC servo controllers detected : ' +
                               str(detected) + '. Check that it is switched on and that Kinesis (or another Python kernel) does not use it.') from e

        self.device.WaitForSettingsInitialized(10000)
        self.device.StartPolling(250)
        time.sleep(0.25)
        self.device.EnableDevice()
        time.sleep(0.25)

        # the TDC001 does not know the actuator : load the settings of the model from the Kinesis database
        config = self.device.LoadMotorConfiguration(SN, DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings)
        config.DeviceSettingsName = model
        config.UpdateCurrentConfiguration()
        self.device.SetSettings(self.device.MotorDeviceSettings, True, False)

        self.model = model
        self.serial_number = SN
        info = self.device.GetDeviceInfo()
        print('stage of the collection objective Thorlabs ' + model + ' (' + str(info.Name) + '), SN = ' + SN + ' connected')

        if home:
            self.home()
        elif not self.is_homed():
            print('Warning, the stage of the collection objective is not homed : run home() to get absolute positions')


    def is_homed(self) -> bool:

        return bool(self.device.Status.IsHomed)


    def home(self):
        """Move the stage to its 0 mm limit switch and set it as the origin."""
        print('homing of the stage of the collection objective...')
        self.device.Home(TIMEOUT_MOVE_MS)
        print('stage of the collection objective homed')


    def move_to(self, position: float, verbose: bool = False, wait: bool = True):
        """
        Move the stage to an absolute position

        Parameters
        ----------
        position : float
            the target position in mm, inside the travel range TRAVEL_MM.
        wait : bool
            if False, returns at once without waiting for the end of the move. The default is True.
        """
        if not TRAVEL_MM[0] <= position <= TRAVEL_MM[1]:
            raise ValueError('position = ' + str(position) + ' mm out of the travel range ' + str(TRAVEL_MM) + ' mm')
        self.device.MoveTo(Decimal(position), TIMEOUT_MOVE_MS if wait else 0)
        if verbose and wait:
            self.read_position(verbose = True)


    def move_by(self, distance: float, verbose: bool = False):
        """
        Move the stage relatively to its current position

        Parameters
        ----------
        distance : float
            the displacement in mm (> 0 or < 0).
        """
        self.move_to(self.read_position() + distance, verbose = verbose)


    def read_position(self, verbose: bool = False) -> float:
        """Return the position of the stage in mm."""
        position = Decimal.ToDouble(self.device.Position)
        if verbose:
            print('stage of the collection objective : position = ' + str(round(position, 4)) + ' mm')
        return position


    def set_velocity(self, velocity: float, acceleration: float = None):
        """
        Set the velocity parameters of the moves

        Parameters
        ----------
        velocity : float
            the maximum velocity in mm/s (Z825B : 2.6 mm/s max).
        acceleration : float
            the acceleration in mm/s². If None, the current one is kept.
        """
        params = self.device.GetVelocityParams()
        if acceleration is None:
            acceleration = Decimal.ToDouble(params.Acceleration)
        self.device.SetVelocityParams(Decimal(velocity), Decimal(acceleration))


    def stop(self):

        self.device.StopImmediate()


    def _approach(self, position: float):
        """Move to position from below (same direction as the scans of the autofocus : backlash)."""
        if position - BACKLASH_MM > TRAVEL_MM[0] and self.read_position() > position - BACKLASH_MM:
            self.move_to(position - BACKLASH_MM)
        self.move_to(position)


    def _scan(self, cam, center, half_range, step, label, stop_event, report):
        """Measure the sharpness of the images of cam at the positions center +/- half_range (increasing)."""
        low = max(TRAVEL_MM[0], center - half_range)
        high = min(TRAVEL_MM[1], center + half_range)
        positions = np.arange(low, high + step / 2, step)
        values = np.zeros(len(positions))
        self._approach(float(positions[0]))
        for i, position in enumerate(positions):
            if stop_event is not None and stop_event.is_set():
                raise AutofocusStopped()
            self.move_to(float(position))
            values[i] = sharpness(cam.fresh_frame())
            report(f'autofocus {label} {i + 1}/{len(positions)} : {position:.3f} mm, sharpness = {values[i]:.4g}')
        return positions, values


    def autofocus(self, cam, half_range: float = None, step: float = None, stop_event = None, status = None) -> float:
        """
        Focus the collection objective : the stage is moved to the position where the image of cam is the sharpest.

        A coarse scan of +/- half_range around the current position (step), extended (AF_MAX_EXTENSIONS times max)
        if the best sharpness is at its edge, then a fine scan of +/- step around the
        best position (step / 5), the maximum of the sharpness is refined by a parabola through the 3 best points.
        If display() of cam is running in another thread, its frames are used, otherwise frames are acquired.
        If the autofocus fails or is stopped, the stage goes back to its starting position.

        Parameters
        ----------
        cam : AndorCam
            the camera giving the images (the spatial camera, flipping mirror on the spatial arm).
        half_range, step : float
            (mm) the range and the step of the coarse scan. The default is None : self.af_half_range, self.af_step.
        stop_event : threading.Event
            the autofocus stops when it is set. The default is None.
        status : function
            called with the text of each step (display in a window). The default is None.

        Returns
        -------
        position : float
            the position of the best focus (mm).
        """
        def report(text):
            print(text)
            if status is not None:
                status(text)

        half_range = self.af_half_range if half_range is None else half_range
        step = self.af_step if step is None else step
        if not self.is_homed():
            raise RuntimeError('the stage of the collection objective is not homed : run home() before the autofocus')

        start = self.read_position()
        try:
            positions, values = self._scan(cam, start, half_range, step, 'coarse', stop_event, report)
            best = positions[np.argmax(values)]
            # best sharpness at the edge of the scan : the focus may be beyond, new scans centered on it
            for extension in range(AF_MAX_EXTENSIONS + 1):
                if np.argmax(values) not in (0, len(values) - 1) or not TRAVEL_MM[0] < best < TRAVEL_MM[1]:
                    break
                if extension == AF_MAX_EXTENSIONS:
                    print('Warning, autofocus : best sharpness still at the edge of the scan, the focus may be farther')
                    break
                positions, values = self._scan(cam, best, half_range, step, 'coarse (extended)', stop_event, report)
                best = positions[np.argmax(values)]

            positions, values = self._scan(cam, best, step, step / 5, 'fine', stop_event, report)
            i = int(np.argmax(values))
            best = positions[i]
            if 0 < i < len(values) - 1:   # vertex of the parabola through the 3 best points
                curvature = values[i - 1] - 2 * values[i] + values[i + 1]
                if curvature < 0:
                    best = best + (step / 5) * (values[i - 1] - values[i + 1]) / (2 * curvature)
            best = float(np.clip(best, TRAVEL_MM[0], TRAVEL_MM[1]))
            self._approach(best)
        except AutofocusStopped:
            self._approach(start)
            report(f'autofocus stopped, back to the starting position {start:.3f} mm')
            return start
        except Exception:
            self._approach(start)
            print(f'autofocus failed, back to the starting position {start:.3f} mm')
            raise

        report(f'autofocus done : {best:.4f} mm (start {start:.4f} mm)')
        return best


    def save_position(self):
        """Write the current position (mm) in LAST_POSITION_FILE."""
        if not self.is_homed():
            print('Warning, the stage of the collection objective is not homed : position not saved in ' + LAST_POSITION_FILE)
            return
        position = self.read_position()
        with open(LAST_POSITION_FILE, 'w') as f:
            f.write(str(position))
        print('position of the stage of the collection objective (' + str(round(position, 4)) + ' mm) saved in ' + LAST_POSITION_FILE)


    def go_to_last_position(self, verbose: bool = False):
        """Move the stage to the position saved in LAST_POSITION_FILE at the last disconnection."""
        if not os.path.isfile(LAST_POSITION_FILE):
            print('Warning, no saved position of the stage of the collection objective (' + LAST_POSITION_FILE + ' not found) : stage not moved')
            return
        with open(LAST_POSITION_FILE, 'r') as f:
            position = float(f.read().strip())
        if not TRAVEL_MM[0] <= position <= TRAVEL_MM[1]:
            raise ValueError('saved position = ' + str(position) + ' mm out of the travel range ' + str(TRAVEL_MM) + ' mm (' + LAST_POSITION_FILE + ')')
        if not self.is_homed():
            raise RuntimeError('the stage of the collection objective is not homed : run home() before going to the saved position')
        self.move_to(position, verbose = verbose)


    def disconnect(self, go_home: bool = True):
        """
        disconnect the stage of the collection objective

        Parameters
        ----------
        go_home: bool.
            home the stage (0 mm) before disconnection. The default is True.
            The position is saved in LAST_POSITION_FILE before.
        """
        self.save_position()
        if go_home:
            self.home()

        self.device.StopPolling()
        self.device.Disconnect()
        self.device = None
        print('Stage of the collection objective disconnected')
