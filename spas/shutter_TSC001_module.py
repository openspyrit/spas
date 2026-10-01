# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:38:24 2026

@author: mahieu
"""

import clr
import sys
import time
from System import Enum

sys.path.append(r"C:/Program Files/Thorlabs/Kinesis")

clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.SolenoidCLI")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.TCube.SolenoidCLI import TCubeSolenoid

class ThorlabsShutter:
    """Class that controls the Thorlabs shutter with the T-Cube solenoid controller TSC001 (Kinesis).

    Example:
        shutter = ThorlabsShutter()
        shutter.init(model = 'TSC001', SN = '85855593')
        shutter.open()
        shutter.close()
        shutter.disconnect()
    """

    def __init__(self):

        self.device = None
        self.model = None
        self.serial_number = None
        self.enum_type = None


    def init(self, model: str = 'TSC001', SN: str = ''):
        """
        Initialize the shutter

        Parameters
        ----------
        model : str
            the model of the controller, compared (case insensitive) to the name read from the device.
            The default is 'TSC001'.
        SN : str
            the serial number of the controller.
        """
        DeviceManagerCLI.BuildDeviceList()
        try:
            self.device = TCubeSolenoid.CreateTCubeSolenoid(SN)
            self.device.Connect(SN)
        except Exception as e:
            detected = list(DeviceManagerCLI.GetDeviceList(TCubeSolenoid.DevicePrefix))
            self.device = None
            raise RuntimeError('shutter Thorlabs, SN = ' + SN + ' not connected (' + str(e) + '). Solenoid controllers detected : ' +
                               str(detected) + '. Check that it is switched on and that Kinesis (or another Python kernel) does not use it.') from e

        info = self.device.GetDeviceInfo()
        device_name = str(info.Name) + ' ' + str(info.Description)
        if model.lower() not in device_name.lower():
            print('Warning, the shutter controller SN = ' + SN + ' is a "' + device_name + '", not a ' + model)

        self.model = model
        self.serial_number = SN
        print('shutter Thorlabs ' + model + ', SN = ' + SN + ' connected')

        time.sleep(0.5)

        self.device.StartPolling(250)
        self.device.EnableDevice()

        time.sleep(0.5)

        # récupérer le type Enum attendu par SetOperatingState
        self.enum_type = self.device.GetOperatingState().GetType()
        

    def open(self):

        active = Enum.ToObject(self.enum_type, 1)
        self.device.SetOperatingState(active)
        
    def close(self):

        self.device.SetOperatingState(self.device.GetOperatingState().Inactive)
    
        time.sleep(0.1)
        
    def state(self):
        
        state = self.device.GetOperatingState()
        if str(state) == '0':
            shutter_state = 'closed'
        elif str(state) == 'Active':
            shutter_state = 'opened'
        else:
            shutter_state = '?'
        print('shutter state = ' + str(shutter_state))
        return str(state)

    def disconnect(self):

        self.device.StopPolling()
        self.device.Disconnect()
        self.device = None
        print('Shutter disconnected')














        
        














