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

#set interface data rate
interface_data_rate = cam_spat.get_limit_bandwidth()
camera_data_rate = int(interface_data_rate / CAMERAS_ON_SAME_CONTROLLER)

#set data rate
cam_spat.set_limit_bandwidth(camera_data_rate)
cam_spec.set_limit_bandwidth(camera_data_rate)

#print device serial numbers
SN_cam_spat = cam_spat.get_device_sn()
SN_cam_spec = cam_spec.get_device_sn()
print('Spatial camera serial number: ' + str(SN_cam_spat))
print('Spectral camera serial number: ' + str(SN_cam_spec))

#settings
cam_spat.set_exposure(2000)
print('cam_spat: Exposure was set to %i us' %cam_spat.get_exposure())
cam_spec.set_exposure(1000)
print('cam_spec: Exposure was set to %i us' %cam_spec.get_exposure())

# set the image data format of the spatial cam as a color
print('data format = ' + cam_spat.get_imgdataformat())
cam_spat.set_imgdataformat('XI_RGB24')#'XI_RGB32')#'XI_RGB_PLANAR')#
print('data format = ' + cam_spat.get_imgdataformat())

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
    data_spec = img_spec.get_image_data_numpy()
    end = time.time()
    
print(save_image + ' elapse time : ' + str(end - start))

#%%stop data acquisition
print('cam_spat: Stopping acquisition...')
cam_spat.stop_acquisition()
print('cam_spec: Stopping acquisition...')
cam_spec.stop_acquisition()
#%%print image data and metadata
print('cam_spat: image (' + str(img_spat.width) + 'x' + str(img_spat.height) + ') received from camera.')
print('First 10 pixels: ' + str(data_spat[:10]) + '\n')

# print('cam_spec: image (' + str(img_spec.width) + 'x' + str(img_spec.height) + ') received from camera.')
# print('First 10 pixels: ' + str(data_spec[:10]))
# print('\n')

plt.figure()
plt.imshow(data_spat)
plt.colorbar()
plt.title('spatial cam')

plt.figure()
plt.imshow(data_spec)
plt.colorbar()
plt.title('spectral cam')

#stop data acquisition
print('cam_spat: Stopping acquisition...')
cam_spat.stop_acquisition()
print('cam_spec: Stopping acquisition...')
cam_spec.stop_acquisition()

#%% find max
import numpy as np

def set_zero(sample, d, val):
    """Set all max value along dimension d in matrix sample to value val."""
    argmax_idxs = sample.argmax(d)
    idxs = [np.indices(argmax_idxs.shape)[j].flatten() for j in range(len(argmax_idxs.shape))]
    idxs.insert(d, argmax_idxs.flatten())
    sample[idxs] = val
    return sample

maxi = np.max(data_spat)
set_zero(data_spat, d = 0, val = 255)
#%% stop communication
cam_spat.close_device()
cam_spec.close_device()

print('cameras closed.')



