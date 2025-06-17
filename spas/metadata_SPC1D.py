# -*- coding: utf-8 -*-
__author__ = 'Guilherme Beneti Martins'

"""Metadata classes and utilities.

Metadata classes to keep and save all relevant data during an acquisition.
Utility functions to recreate objects from JSON files, save them to JSON and to
improve readability.
"""

# import json  
# from datetime import datetime
# from enum import IntEnum
# from dataclasses import dataclass, InitVar, field
# from typing import Optional, Union, List 
from typing import Tuple
from pathlib import Path
# import os
# from dataclasses_json import dataclass_json
# import numpy as np
# import ctypes as ct
# import pickle
# ##### DLL for the DMD
# try:
#     import ALP4
# except: # in the cas the DLL of the DMD is not installed
#     class ALP4:
#         pass
#     setattr(ALP4, 'ALP4',  None)
#     print('DLL of the DMD not installed')
# ##### DLL for the spectrometer Avantes 
# try:
#     from msl.equipment.resources.avantes import MeasConfigType
# except: # in the cas the DLL of the spectrometer is not installed
#     class MeasConfigType:
#         pass
#     MeasConfigType =  None
#     print('DLL of the spectrometer not installed !!!')

# ##### DLL for the camera
# try:
#     from pyueye import ueye
#     dll_pyueye_installed = 1
# except:
#     dll_pyueye_installed = 0
#     print('DLL of the cam not installed !!')
    



  
