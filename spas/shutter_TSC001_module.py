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

    def __init__(self, serial):

        DeviceManagerCLI.BuildDeviceList()

        self.device = TCubeSolenoid.CreateTCubeSolenoid(serial)
        self.device.Connect(serial)
        print('shutter connected')
        
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
        print('Shutter disconnected')














        
        














