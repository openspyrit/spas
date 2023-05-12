# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 17:21:21 2022

@author: mahieu
"""

from matplotlib import pyplot as plt
import cv2
import numpy as np
import pickle
from scipy.fft import fft, fftfreq

######################## INPUT ###############################
data_folder_name = '2022-10-19_lamb_brain_motion_plus_PpIX_location_3'
data_name    = 'Motion_2.5mm_WhiteLight_ON_BlueLaser_OFF_PpIX_NO_im_64x64_Zoom_x1_ti_20ms'
video_format = 'avi'
##############################################################
filename = '../data/' + data_folder_name + '/' + data_name + '/' + data_name + '_video.' + video_format
filename_out = '../data/' + data_folder_name + '/' + data_name + '/' + data_name + '_video_fps.' + video_format
if filename.find('.avi') > 1:
    vid_format = 'avi'
elif filename.find('.bin') > 1:
    vid_format = 'raw'
else:
    vid_format = 'unknown'
    print('video format unknown')
################################### read metadata cam #######
data_path = '../data/' + data_folder_name + '/' + data_name + '/' + data_name
cam_metadata_path = data_path + '_metadata_cam.pkl'
file = open(cam_metadata_path,'rb')
cam_metadata = pickle.load(file)
file.close()

tti = cam_metadata['time_array']
gate_period = cam_metadata['gate_period']
ti = tti[:len(tti)//gate_period]
delta_ti = (ti[-1]-ti[0])/len(ti)
real_fps = 1/delta_ti
#%%############################### RAW Video ####################################
if vid_format == 'raw': 
    ######## extract video header ##################
    file = open(filename,"rb")
    data_header = np.fromfile(file, dtype="uint16")
    file.close()
    video_header = 40
    frame_header = 48
    video_header_str = data_header[1:video_header]
    
    ######## extract video features ############## 
    Width = np.uint32(video_header_str[9])
    Height = np.uint32(video_header_str[11])
    frame_nbr = np.uint32(video_header_str[15])
    fps = float()
    
    ######## read frame header ##############
    frame_header_str = []
    for ii in range(frame_nbr, 0, -1):
        B = np.arange(start = Width*Height*(ii-1), stop = Width*Height*(ii-1)+frame_header, step = 1, dtype=np.uint32)
        frame_header_str.append(B)
        
    ###### read real data, take off header of the video and for each frame ########
    file = open(filename,"rb")
    dat = np.fromfile(file, dtype="uint8")
    file.close()
    ## ectract video header ##
    dat3 = dat[video_header:]
    ## extract frame header ##   
    #frame_header_str = []
    for ii in range(frame_nbr, 0, -1):
        B = np.arange(start = Width*Height*(ii-1), stop = Width*Height*(ii-1)+frame_header, step = 1, dtype=np.uint32)
        #frame_header_str.append(B)
        dat3 = np.delete(dat3,[B])
    
    ###### reshape matrix #################
    mat = np.reshape(dat3, (frame_nbr, Height, Width))
    
    ## plot indiviual frames
    plot_nbr = frame_nbr
    if frame_nbr >= 20:
        plot_nbr = 20
            
    for ii in range(plot_nbr):
        plt.figure()
        plt.imshow(mat[ii,:,:])
        plt.colorbar();
        plt.title('frame = ' + str(ii))


############################### AVI Video ####################################
elif vid_format == 'avi':
    cap = cv2.VideoCapture(filename)
    
    Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    Width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    fps = cap.get(cv2.CAP_PROP_FPS) # float `fps`
    frame_nbr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # float `total_frame_in_the_vid
    # out = cv2.VideoWriter(filename_out,cv2.VideoWriter_fourcc('M','J','P','G'), real_fps, (Height,Width))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename_out, fourcc, real_fps, (Width, Height))
    vid = np.zeros((Height, Width, frame_nbr))
    
    for ii in range(frame_nbr):
        ret, frame = cap.read()
        vid[:,:,ii] = frame[:,:,0]
        if ret == True:
            out.write(frame)
            if ii <= 15 or (ii >= frame_nbr-10 and ii <= frame_nbr-1):
                plt.figure()
                plt.imshow(frame[:,:,0])
                plt.colorbar();
                plt.title('frame = ' + str(ii))
                #plt.show()
            
    
    
    cap.release()
    cv2.destroyAllWindows()
    
    

print('')
print('height:', Height)
print('width:', Width)
print('fps:', fps)  # float `fps`
print('frames count:', frame_nbr)  # float `frame_count`
#%% exploit video from avi file
mean_2D = np.mean(np.mean(vid, axis = 1), axis=0)
deb = 100#0#
fin = 300#4095#

plt.figure()
plt.plot(ti[deb:fin], mean_2D[deb:fin])
plt.xlabel('s')
plt.title('ALL')
plt.grid()
plt.savefig(data_path + '_time_ALL.png')
##### FFT ######
mean_2D = mean_2D - np.mean(mean_2D)
yf = fft(mean_2D)
xf = fftfreq(frame_nbr, delta_ti)[:frame_nbr//2]

plt.figure()
plt.plot(xf, 2.0/frame_nbr * np.abs(yf[0:frame_nbr//2]))
plt.title('ALL')
plt.grid()
plt.savefig(data_path + '_fft_ALL.png')

#%% plot part of the video
zone = np.array([[230,240,320,330],[250,300,480,520],[350,400,225,375],[230,230,320,320],[375,375,350,350]], np.int16)

for ii in range(len(zone)):
    fov = np.mean(np.mean(vid[zone[ii,:],:], axis = 1), axis=0)

    plt.figure()
    plt.plot(ti[deb:fin], fov[deb:fin])
    plt.xlabel('s')
    
    if (zone[ii,0] == zone[ii,1])  & (zone[ii,2] == zone[ii,3]) == 1:
        plt.title('pixel [' + str(zone[ii,0]) + ',' + str(zone[ii,2]) + ']')
        plt.grid()
        plt.savefig(data_path + '_time_pixel_[' + str(zone[ii,0]) + ',' + str(zone[ii,2]) + '].png')
    else:
        plt.title('zone [' + str(zone[ii,0]) + ':' + str(zone[ii,1]) + ',' + str(zone[ii,2]) + ':' + str(zone[ii,3]) + ']')
        plt.grid()
        plt.savefig(data_path + '_time_zone_[' + str(zone[ii,0]) + '-' + str(zone[ii,1]) + ',' + str(zone[ii,2]) + '-' + str(zone[ii,3]) + '].png')
##### FFT ######
    fov = fov - np.mean(fov)
    yf = fft(fov)
    xf = fftfreq(frame_nbr, delta_ti)[:frame_nbr//2]
    
    plt.figure()
    plt.plot(xf, 2.0/frame_nbr * np.abs(yf[0:frame_nbr//2]))
    if (zone[ii,0] == zone[ii,1])  & (zone[ii,2] == zone[ii,3]) == 1:
        plt.title('pixel [' + str(zone[ii,0]) + ',' + str(zone[ii,2]) + ']')
        plt.grid()
        plt.savefig(data_path + '_fft_pixel_[' + str(zone[ii,0]) + ',' + str(zone[ii,2]) + '].png')
    else:
        plt.title('zone [' + str(zone[ii,0]) + ':' + str(zone[ii,1]) + ',' + str(zone[ii,2]) + ':' + str(zone[ii,3]) + ']')
        plt.grid()
        plt.savefig(data_path + '_fft_zone_[' + str(zone[ii,0]) + '-' + str(zone[ii,1]) + ',' + str(zone[ii,2]) + '-' + str(zone[ii,3]) + '].png')


























  
       