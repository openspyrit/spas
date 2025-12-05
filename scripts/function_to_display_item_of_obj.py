# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 10:44:29 2025

@author: admin
"""

# import pprint as pp

# pp.pprint(cam_spat)

# 300 800 get
# 996 1400 set

def dump(obj, incd, incf):
    inc = 0
    for attr in dir(obj):
        inc = inc + 1
        if inc > incd and inc < incf:
            print("obj.%s = %r" % (attr, getattr(obj, attr)))