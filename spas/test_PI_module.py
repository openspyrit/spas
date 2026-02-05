# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 12:46:21 2026

@author: equipe-onli
"""

# import pipython
# from pipython import GCSDevice, pitools

from spas.PI_module import init_PI, disconnect_stage, read_position, move_to_middle, go_to_zero, move_an_axis, PIJogControl
import numpy as np

pidevice, stage_tools = init_PI(Model = 'C-884', SN = '0000000000', verbose = True)

poistion = read_position(pidevice, stage_tools, verbose = True)

move_to_middle(pidevice, stage_tools)

array_to_move = np.linspace(1, 10, 10, endpoint=True)
move_an_axis(pidevice, stage_tools, axes = ['2'], array_to_move = array_to_move, verbose = True)

pidevice.MOV('3', 5)

go_to_zero(pidevice, stage_tools)





disconnect_stage(pidevice, stage_tools)










pidevice = GCSDevice('C-884')

try:
    pidevice.ConnectUSB(serialnum = '0000000000')
    pidevice.qIDN()
    print('PI tanslation stage connected')
    axes = pitools.getaxeslist(pidevice, None)
    # print(axes)
    Min_Max_axes = [ []*2 for i in range(3)]
    for i in range(1,4):
        # min travel range
        minrange_dict = pitools.getmintravelrange(pidevice, i)
        minrange = list(minrange_dict.values())[0]
        # print('min travel range = ' + str(minrange) + ' mm')
        # max travel range
        maxrange_dict = pitools.getmaxtravelrange(pidevice, i)
        maxrange = list(maxrange_dict.values())[0]
        # print('max travel range = ' + str(maxrange) + ' mm')    
        Min_Max_axes[i-1] = [minrange, maxrange]
except:
    print('!!!! Warning, PI tanslation stage not connected !!!!')
    print('Turn on the controller and wait until the both green led are stabilized')
    
    
    
try:
    pidevice.qIDN()
    
    # Get a list of all axes of the controller
    axes = pitools.getaxeslist(pidevice, None)
    # print(axes)
    if len(axes) == 3:
        print('PI tanslation stage connected')
        Min_Max_axes = [ []*2 for i in range(3)]
        for i in range(1,4):
            # min travel range
            minrange_dict = pitools.getmintravelrange(pidevice, i)
            minrange = list(minrange_dict.values())[0]
            # print('min travel range = ' + str(minrange) + ' mm')
            # max travel range
            maxrange_dict = pitools.getmaxtravelrange(pidevice, i)
            maxrange = list(maxrange_dict.values())[0]
            # print('max travel range = ' + str(maxrange) + ' mm')    
            Min_Max_axes[i-1] = [minrange, maxrange]
        
    else:
        print('One or more axes of the PI translation stage are not connected')
except:
    print('!!!! Warning, PI tanslation stage not connected !!!!')




axes_xyz = ['X', 'Y', 'Z']
for i in range(1,4): 
    # Current position
    pos_dict = pidevice.qPOS(i)
    pos = list(pos_dict.values())[0]
    print('Current position of ' + axes_xyz[i-1] + ' = ' + str(pos) + ' mm')


#absolute move
pidevice.MOV('3', 5)



# go to zero
for i in range(1,4):
    pidevice.MOV(i, 0)

# Emergency stop
pitools.stopall(pidevice)






for ax in axes:
    if ax <= 2:
        pidevice.SVO(ax, 1)
    else:
        pidevice.SVO(ax, 0)

print("Servo:", pidevice.qSVO())

# 2) Définir la position actuelle comme référence logicielle
for ax in axes:
    # pidevice.DFH(ax)

# ou si DFH n’existe pas :
    pidevice.POS(ax, 0)

# 3) Vérifier que les axes sont maintenant considérés référencés
print("Référencé:", pidevice.qFRF())



for ax in axes:
    pidevice.SVO(ax, 1)

# 2) Autoriser référence logicielle
for ax in axes:
    pidevice.RTO(ax)

# 3) Vérifier
print("Référencé:", pidevice.qFRF())



for ax in axes:
    pidevice.SPA(ax, 0x0F00, 1)  # force logical reference



print(pidevice.qFRF())




pitools.startup(pidevice, ['M-111.1DG', 'M-111.1DG', None], ['FNL', None, 'FPL'], [True, True, False])


# pitools.startup(pidevice, None, 'FRF', None, [0x2, 0x2, 0x4, ])


pitools.setservo(pidevice, [1, 2, 3], [True, False, False])
print("Servo:", pidevice.qSVO())

# pitools.startup(pidevice, None, 'FRF', [True, None, False, ])






