from ximea import xiapi
from matplotlib import pyplot as plt
import time

CAMERAS_ON_SAME_CONTROLLER = 2

#create instance for cameras
cam_spat = xiapi.Camera(dev_id=0)
cam_spec = xiapi.Camera(dev_id=1)

#start communication
print('Opening cameras...')
# cam_spat.open_device()
# cam_spec.open_device()

# or open cam by its SN
cam_spat.open_device_by_SN('BRCID2503000')
cam_spec.open_device_by_SN('BRMID2503000') 

#print device serial numbers
SN_cam_spat = cam_spat.get_device_sn()
SN_cam_spec = cam_spec.get_device_sn()
print('Spatial camera serial number: ' + str(SN_cam_spat))
print('Spectral camera serial number: ' + str(SN_cam_spec))

#%% setup cam
#set interface data rate
# interface_data_rate = cam_spat.get_limit_bandwidth()
interface_data_rate = cam_spat.get_limit_bandwidth_maximum()
camera_data_rate = int(interface_data_rate / CAMERAS_ON_SAME_CONTROLLER)

#set data rate
cam_spat.set_limit_bandwidth(camera_data_rate)
cam_spec.set_limit_bandwidth(camera_data_rate)

print('bandwidth cam spat = ' + str(cam_spat.get_limit_bandwidth()))
print('bandwidth cam spec = ' + str(cam_spec.get_limit_bandwidth()))

# get data format
data_format_spat = cam_spat.get_imgdataformat()
data_format_spec = cam_spec.get_imgdataformat()

# set data format
cam_spat.set_imgdataformat('XI_RGB48')
# cam_spec.set_imgdataformat('XI_RAW16')


# get bit depth
sensor_bit_depth_spat = cam_spat.get_sensor_bit_depth()
sensor_bit_depth_spec = cam_spec.get_sensor_bit_depth()


# get output data format
output_bit_depth_spat = cam_spat.get_output_bit_depth()
# output_bit_depth_spec = cam_spec.get_output_bit_depth()

# set output data format
cam_spat.set_output_bit_depth('XI_BPP_10') 
# cam_spec.set_output_bit_depth('XI_BPP_10') 


# get image data bit depth
image_data_bit_depth_spat = cam_spat.get_image_data_bit_depth()
image_data_bit_depth_spec = cam_spec.get_image_data_bit_depth()

# set image data bit depth
cam_spat.set_image_data_bit_depth('XI_BPP_16') # 'XI_BPP_8'
# cam_spec.set_image_data_bit_depth('XI_BPP_10')


# get output data packing
output_bit_packing_spat = cam_spat.is_output_bit_packing() 
output_bit_packing_spec = cam_spec.is_output_bit_packing() 

# enable output data packing
if output_bit_packing_spat == False:
    cam_spat.enable_output_bit_packing()
    print('enable spat cam to output bit packing')
# if output_bit_packing_spec == False:
#     cam_spec.enable_output_bit_packing()
#     print('enable spec cam to output bit packing')

#settings
cam_spat.set_exposure(500)
print('cam_spat: Exposure was set to %i us' %cam_spat.get_exposure())
cam_spec.set_exposure(1200)
print('cam_spec: Exposure was set to %i us' %cam_spec.get_exposure())

# set gain
gain = 18
cam_spec.set_gain(gain)



# set image output bit depth
cam_spec.set_sensor_bit_depth('XI_BPP_10')
print('sensor bit depth = ' + cam_spec.get_sensor_bit_depth())

cam_spec.set_imgdataformat('XI_RAW16')
print('sensor bit depth = ' + cam_spec.get_imgdataformat())

cam_spec.set_output_bit_depth('XI_BPP_10') 
print('sensor bit depth = ' + cam_spec.get_output_bit_depth())

# cam_spec.set_output_bit_depth('XI_BPP_16')
# print('image output bit depth = ' + cam_spec.get_output_bit_depth())





# set the image data format of the spatial cam as a color
print('data format = ' + cam_spat.get_imgdataformat())
cam_spat.set_imgdataformat('XI_RGB24')#'XI_RGB32')#'XI_RGB_PLANAR')#
print('data format = ' + cam_spat.get_imgdataformat())


cam_spec.set_buffer_policy('XI_BP_SAFE')
cam_spec.set_acq_buffer_size(int(cam_spec.get_acq_buffer_size_maximum()/4))
print('buffer size spec = ' + str(cam_spec.get_acq_buffer_size()))

cam_spec.set_buffers_queue_size(cam_spec.get_buffers_queue_size_maximum())
buffers_queue_size = cam_spec.get_buffers_queue_size()
print('buffers queue size spec  = ' + str(buffers_queue_size))


#create instance of Image to store image data and metadata
img_spat = xiapi.Image()
img_spec = xiapi.Image()

#%% specify the format of the acquired data
save_image = 'np_array'# 'raw'#

#start data acquisition
print('Starting data acquisition...\n')
cam_spat.start_acquisition()
cam_spec.start_acquisition()

#get data and pass them from cameras to img
cam_spat.get_image(img_spat)
cam_spec.get_image(img_spec)

#get raw data from cameras
#for Python2.x function returns string
#for Python3.x function returns bytes
#%%
if save_image == 'raw':
    start = time.time()
    data_raw_spat = img_spat.get_image_data_raw()
    data_raw_spec = img_spec.get_image_data_raw()
    
    #transform data to list
    data_spat = list(data_raw_spat)
    data_spec = list(data_raw_spec)
    end = time.time()
elif save_image == 'np_array':
    start = time.time()
    data_spat = img_spat.get_image_data_numpy()
    data_spec = img_spec.get_image_data_numpy(invert_rgb_order = True)
    end = time.time()
    
print(save_image + ' elapse time : ' + str(end - start))     
#%% trhead
def runCam_thread_little(cam_spec, all_path):

    img_spec = xiapi.Image()
    cam_spec.start_acquisition()
    i = 0
    print('i = ' + str(i))
    while True:
        cam_spec.get_image(img_spec)
        data_spec = img_spec.get_image_data_numpy(invert_rgb_order = True)
        outpath = 'test1/spectral_' + str(i)
        np.savez(outpath, data_spec)
        i = i + 1
        print('i = ' + str(i))
        if i > 100:
            print('i = ' + str(i))
            cam_spec.stop_acquisition()
            break
#%% loop acquisition
import numpy as np
import threading

# cam_spec.set_trigger_source('XI_TRG_OFF')
# cam_spec.set_trigger_software(1)

# gpi_mode = 'XI_GPI_TRIGGER'
# cam_spec.set_gpi_mode(gpi_mode)
# trigger_source = 'XI_TRG_EDGE_RISING'
# cam_spec.set_trigger_source(trigger_source)
# trigger_selector = 'XI_TRG_SEL_FRAME_START'
# trigger_selector = 'XI_TRG_SEL_EXPOSURE_START'
# cam_spec.set_trigger_selector(trigger_selector)

cam_spec.set_gpi_mode('XI_GPI_TRIGGER')
# cam_spec.set_gpo_mode('XI_GPO_HIGH_IMPEDANCE') 
cam_spec.set_trigger_source('XI_TRG_EDGE_RISING')
trigger_selector = 'XI_TRG_SEL_EXPOSURE_ACTIVE'
trigger_selector = 'XI_TRG_SEL_FRAME_BURST_START'

trigger_selector = 'XI_TRG_SEL_FRAME_START'
cam_spec.set_trigger_selector(trigger_selector)
cam_spec.set_trigger_overlap('XI_TRG_OVERLAP_OFF')

all_path = 'test/'

time.sleep(1.2)
# time.sleep(0.1)

xx = threading.Thread(target = runCam_thread_little, args=(cam_spec, all_path))
xx.start()

DMD.Run(loop=True)

time.sleep(3)

DMD.Halt()


#%%
from matplotlib import pyplot as plt
import pickle
import numpy as np
import os

fi = 'spectral'#'spatial'#
plot_fig = True
i = 0
data_path_folder = 'test1/'
data_file_list = os.listdir(data_path_folder)
for file in data_file_list:
    if file.startswith(fi):
        data_path = data_path_folder + file
        # data_path = 'test.npz'
        file = np.load(data_path)    
        da = file['arr_0']
        
        if plot_fig == True:    
            if i <= 5:
                img16 = da
                if fi == 'spatial':
                    img8 = (img16/256).astype('uint8')
                    img8 = np.flip(np.flip(img8, axis = 1), axis = 0)
                elif fi == 'spectral':
                    img8 = da
                plt.figure()
                plt.imshow(img8)
                plt.title(i)
                plt.colorbar()
        
        i = i + 1
#%%stop data acquisition
print('cam_spat: Stopping acquisition...')
cam_spat.stop_acquisition()
print('cam_spec: Stopping acquisition...')
cam_spec.stop_acquisition()
# #%%print image data and metadata
# # print('cam_spat: image (' + str(img_spat.width) + 'x' + str(img_spat.height) + ') received from camera.')
# # print('First 10 pixels: ' + str(data_spat[:10]) + '\n')

# # print('cam_spec: image (' + str(img_spec.width) + 'x' + str(img_spec.height) + ') received from camera.')
# # print('First 10 pixels: ' + str(data_spec[:10]))
# # print('\n')

# # plt.figure()
# # plt.imshow(data_spat, color_continuous_scale='RdBu_r', origin='lower')
# # plt.colorbar()
# # plt.title('spatial cam')

# import plotly.express as px
# fig = px.imshow(data_spat, color_continuous_scale='RdBu_r', origin='lower')
# fig.show()

# fig = px.imshow(data_spat, zmin=50, zmax=200)
# fig.show()

# # plt.figure()
# # plt.imshow(data_spec)
# # plt.colorbar()
# # plt.title('spectral cam')

# #stop data acquisition
# print('cam_spat: Stopping acquisition...')
# cam_spat.stop_acquisition()
# print('cam_spec: Stopping acquisition...')
# cam_spec.stop_acquisition()

#%% plot
plt.figure()
plt.imshow(data_spec)
plt.colorbar()

#%% find max
# import numpy as np

# def set_zero(sample, d, val):
#     """Set all max value along dimension d in matrix sample to value val."""
#     argmax_idxs = sample.argmax(d)
#     idxs = [np.indices(argmax_idxs.shape)[j].flatten() for j in range(len(argmax_idxs.shape))]
#     idxs.insert(d, argmax_idxs.flatten())
#     sample[idxs] = val
#     return sample

# maxi = np.max(data_spat)
# set_zero(data_spat, d = 0, val = 255)
#%% stop communication
cam_spat.close_device()
cam_spec.close_device()

print('cameras closed.')



