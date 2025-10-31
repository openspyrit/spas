import numpy as np
from ximea import xiapi
import PIL.Image
import ximea
import time
import sys


# user parameters
exposure = 1400
wanted_frames=5

print("used api version " + ximea.__version__)
print("used python version " +sys.version)

try:
    camera = xiapi.Camera()
    camera.set_debug_level('XI_DL_DISABLED')
    camera.open_device()
except xiapi.Xi_error as e:
    raise Exception('Could not connect to camera.') from e

except KeyboardInterrupt:
    print('\nThe process has been stopped')
    

print('-----camera connected successfully-----')
camera.set_exposure(exposure) #set exposure in us
camera.set_sensor_bit_depth('XI_BPP_10')
camera.set_output_bit_depth('XI_BPP_10')
camera.set_buffer_policy('XI_BP_SAFE')
camera.set_limit_bandwidth(camera.get_limit_bandwidth_maximum())

camera.set_trigger_source('XI_TRG_SOFTWARE') #triggerred by software

camera.set_trigger_selector('XI_TRG_SEL_FRAME_BURST_START') # burst regime
camera.set_acq_buffer_size(camera.get_acq_buffer_size_maximum()) #set buffer to maximum
camera.set_buffers_queue_size(camera.get_buffers_queue_size_maximum())
camera.set_acq_frame_burst_count(wanted_frames) #set the number of frames to be acquired after trigger pulse has been sent to the camera
image = xiapi.Image() 
time_stmp = 0

#start data acquisition
print('Starting data acquisition...\n')
camera.start_acquisition()
input('******Press enter to start capturing******')
camera.set_trigger_software(1)

for n in range(wanted_frames):
    camera.get_image(image, timeout=int(5000))
    data = image.get_image_data_numpy()
    img = PIL.Image.fromarray(data, 'L')
    time_stmp = (image.tsSec) + ((image.tsUSec)/1000000)
    img.save("burstmode_image_"+ str(n+1) +"_" + str(time_stmp) + ".BMP")
    print('Time Stamp:{}s' .format(time_stmp))
    if n != 0:
        print('fps:{}'.format(1/(time_stmp-lastTime)))
    lastTime = time_stmp
        

print('\n-----------COUNTERS-----------') # reading counters
camera.set_counter_selector('XI_CNT_SEL_TRANSPORT_SKIPPED_FRAMES')
print('Transport skipped frames: ',camera.get_counter_value())
camera.set_counter_selector('XI_CNT_SEL_API_SKIPPED_FRAMES')
print('API skipped frames: ',camera.get_counter_value())
camera.set_counter_selector('XI_CNT_SEL_TRANSPORT_TRANSFERRED_FRAMES')
print('Transferred frames: ',camera.get_counter_value())
camera.stop_acquisition()
camera.close_device()
print("---device closed---")
