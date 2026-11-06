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
        obj_stage.disconnect(go_home = True)
    """

    def __init__(self):

        self.device = None
        self.model = None
        self.serial_number = None


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


    def move_to(self, position: float, verbose: bool = False):
        """
        Move the stage to an absolute position

        Parameters
        ----------
        position : float
            the target position in mm, inside the travel range TRAVEL_MM.
        """
        if not TRAVEL_MM[0] <= position <= TRAVEL_MM[1]:
            raise ValueError('position = ' + str(position) + ' mm out of the travel range ' + str(TRAVEL_MM) + ' mm')
        self.device.MoveTo(Decimal(position), TIMEOUT_MOVE_MS)
        if verbose:
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
