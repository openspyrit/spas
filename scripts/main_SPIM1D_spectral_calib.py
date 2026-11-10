#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 2026
@author: mahieu

Spectral calibration of the spectral arm of the SPIM (copy of main_SPIM1D_light-sheet_calib.py) :
    - wavelength vector : Hg-Ar lamp Ocean Optics HG-1 (PART A), written in WAVELENGTH_CALIB
      (spas/spectral_calib_SPIM1D.py) and read by define_wavelengths_matrix, right for any Lc, ROI and binning,
    - spectral resolution (FWHM of a line, mean between 380 and 930 nm), measured with one of 2 methods :
        'lamp'  : the lines of the HG-1 (PART A, same acquisition as the wavelength calibration),
        'laser' : the laser lines scattered by the fluorescent cuvette (PART B), with the light sheet of the SPIM.
      The method chosen in resolution_method gives resolution_th of Spectrograph_Parameters, for the slit width.

With a wide slit (3 mm), the resolution is given by the width of the image of the light in the plane of the
slit : the image of the fiber with the lamp, the image of the light sheet with the laser. The laser method
measures the resolution of the acquisitions ; the lamp is needed for the wavelength vector.

PROTOCOL A - HG-1 lamp (wavelength calibration + resolution 'lamp')
    1. Fiber of the HG-1 at the position of the sample, in front of the collection objective (in the holder of
       the cuvette, cuvette removed). Lamp switched on 5 min before (the Hg lines are stable after the warm-up).
    2. Lasers off, shutter closed (cell A1).
    3. Spatial camera (cell A2) : center the fiber on the image (PI stage, or by hand) and focus with the
       collection objective ("collect objective" axis or AUTOFOCUS) : the face of the fiber is sharp.
    4. Spectral camera (cell A3) : lines thin and bright (fine focus of the collection objective).
    5. Cells A4 to A8 : acquisition for each Lc (auto-exposure), fit, check of the figures, writing in
       WAVELENGTH_CALIB, transfer to girder, check with another Lc and binning.

PROTOCOL B - laser line (resolution 'laser', configuration of the acquisitions)
    1. Fluorescent cuvette in its holder, at the position of the sample (as for the acquisitions).
    2. Remove the notch filters (473 and 594 nm) of the collection path, otherwise the laser line is cut,
       or use a laser that is not blocked by the notch filters. Fluorescence of the cuvette : a broad
       background, removed locally under the line.
    3. Choose the lasers (cell B1) : a narrow line (DPSS / OPSS, single mode). The width of the laser line
       (datasheet) is removed quadratically (linewidth) : give it if it is not << the resolution (multimode
       diode : ~1 nm). Low power (~0.5 mW) : the light scattered by the cuvette saturates quickly.
    4. Spatial camera (cell B2) : light sheet in the cuvette, at the position used for the acquisitions,
       focus with the collection objective.
    5. Spectral camera (cell B3) : thinnest laser line (fine focus of the collection objective), no saturation.
    6. Cells B4 to B7 : scan of Lc around each laser (the line moves along the sensor), FWHM of the line,
       writing in WAVELENGTH_CALIB, transfer to girder. Put back the notch filters after the measurement.
"""


# packages
import os
os.chdir('C:\\openspyrit\\spas\\scripts')
import numpy as np
import time
from scipy import signal
from matplotlib import pyplot as plt

from spas.DMD_module import init_DMD, disconnect_DMD, play_one_pattern
from spas.spectro_Shamrock_module import Shamrock, Spectrograph_Parameters
from spas.cam_Andor_module import AndorCam, cam_Parameters
from spas.PI_module import PIStage
from spas.shutter_TSC001_module import ThorlabsShutter
from spas.flipping_mirror_MFF101_module import MFF
from spas.stage_collect_obj_module import CollectObjStage
from spas.combiner_module import Combiner
from spas.transfer_data_to_girder import transfer_data_spectral_calib
from spas.acquisition_SPIM1D import plot_spectrum
import spas.spectral_calib_SPIM1D as sc
from spas.spectral_calib_SPIM1D import define_wavelengths_matrix, spectral_resolution, physical_pixels
#%% Initialize hardware
spectrograph = Shamrock()
spectrograph.init(model = 'andor_shamrock', SN = 'SR-2070')   # Shamrock 500i (focal length 500 mm)
DMD, DMD_initial_memory = init_DMD(dmd_lib_version = '4.3')   # the DMD triggers the cameras (and makes the light sheet)
cam_spat = AndorCam()
cam_spat.init(model = 'ZYLA-4.2P-USB3-S', SN = 'VSC-10323', arm = 'spatial')   # alignment of the fiber or of the light sheet
cam_spec = AndorCam()
cam_spec.init(model = 'ZYLA-4.2P-USB3', SN = 'VSC-23585', arm = 'spectral')
stage = PIStage()
stage.init(model = 'C-884', SN = '0000000000', verbose = True)
shutter = ThorlabsShutter()
shutter.init(model = 'TSC001', SN = '85855593')
mirror = MFF()
mirror.init(model = 'MFF101', SN = '37010810')
obj_stage = CollectObjStage()
obj_stage.init(model = 'Z825B', SN = '83850559', home = True)
combi = Combiner()
combi.init(model = 'combiner', SN = '2')
combi.all_off()
shutter.close()
obj_stage.go_to_last_position(verbose = True)   # collection objective back to its position of the last session
#%% Parameters
resolution_method = 'laser'    # 'lamp' (PART A) or 'laser' (PART B) : the method that gives resolution_th of Spectrograph_Parameters
grating_nbr       = 1          # 300 gr/mm
slit_width        = 3000       # (µm) width of the input slit. Not motorized : stored in Spectrograph_Parameters (slit_width, resolution_th)
setup_version     = 'setup_v1.0'
collection_access = 'public'   #'private'#
data_folder_name  = '2026-11-10_spectral_calib'
pattern           = dict(pattern_to_display = 'gray_0', pattern_dim = '1D', scan_mode = 'Walsh_sparse',
                         Np = 128, pattern_thickness = 16)   # pattern of the DMD : trigger of the cameras (and light sheet with the laser)
#%% setup the Spectrograph
spectrograph_params = spectrograph.setup(grating_nbr = grating_nbr, print_select   = True,   # Arg:  1
                                         position    = 546,         print_position = True,   # the central wavelength of the grating
                                         slit_width  = slit_width,                           # the width of the input slit in (µm), None : not changed
                                         input_port  = None,                                 # 'direct' or 'side', None : not changed
                                         output_port = None,        print_ports    = True)   # 'direct' or 'side', None : not changed
#%% setup Spatial Camera : image of the fiber or of the light sheet, at the place of the sample
cam_spat_params = cam_spat.setup(expos_time  = 0.05,     # (s), decrease it if the image is saturated
                                 gain        = 1,        # 1 or 2
                                 width       = 2048,     # max = 2048
                                 height      = 2048,     # max = 2048
                                 offsetX     = 1,        # 1
                                 offsetY     = 1,        # 1
                                 binningX    = 1,        # int < 2048
                                 binningY    = 1,        # int < 2048
                                 encodPix    = 12,       # 12 or 16 bit
                                 snapshot    = False)    # if false => acquire video, if True => acquire an image
#%% setup Spectral Camera : no binning along the wavelength (binningX = 1) for the calibration
cam_spec_params = cam_spec.setup(expos_time  = 0.05,     # (s), adjusted for each Lc by the auto-exposure
                                 gain        = 1,        # 1 or 2
                                 width       = 2048,     # max = 2048
                                 height      = 2048,     # max = 2048
                                 offsetX     = 1,        # 1
                                 offsetY     = 1,        # 1
                                 binningX    = 1,        # keep 1 : full resolution along the wavelength
                                 binningY    = 8,        # int < 2048
                                 encodPix    = 12,       # 12 or 16 bit
                                 snapshot    = True)     # if false => acquire video, if True => acquire an image
x = physical_pixels(cam_spec_params.width, cam_spec_params.offsetX, cam_spec_params.binningX)   # columns -> pixels of the sensor
try:
    pixel_width = cam_spec.get("PixelWidth")   # (µm)
except Exception:
    pixel_width = 6.5
#%% functions : snapshot of the spectral camera with the auto-exposure
def snap_spec(expos_time):
    """One snapshot of the spectral camera (triggered by the DMD) with the exposure time expos_time (s)."""
    expos_time = min(max(expos_time, cam_spec.get("min_ExposureTime")), cam_spec.get("max_ExposureTime"))
    cam_spec.set("ExposureTime", expos_time)
    params = cam_Parameters(cam = cam_spec)
    play_one_pattern(DMD, DMD_initial_memory, cam_Par = params, **pattern)
    try:
        image = cam_spec.snap()
    finally:
        DMD.Halt()
    # hot pixels removed with a median along the height of the slit only (3 rows x 1 column) :
    # a 3x3 median would distort the profile of the lines, used to measure the resolution
    return signal.medfilt2d(image, kernel_size = (3, 1)), cam_spec.get("ExposureTime")


def snap_spec_auto(expos_time, target = 0.6, expos_max = 2, n_iter = 6):
    """Snapshot with the highest line at ~target x full scale (no saturation)."""
    full_scale = 2**cam_spec.bit_depth() - 1
    for _ in range(n_iter):
        image, expos_time = snap_spec(expos_time)
        top = image.max()
        line = top - np.median(image)
        if top >= 0.95 * full_scale:
            expos_time = expos_time / 4                                    # saturation
        elif line < 0.3 * full_scale and expos_time < expos_max:
            expos_time = min(expos_time * min(target * full_scale / max(line, 1), 10), expos_max)
        else:
            break
    saturated = image.max() >= 0.95 * full_scale   # the FWHM of saturated lines is not used
    if saturated:
        print('Warning, saturation at the minimum exposure time : decrease the power of the light')
    return image, expos_time, saturated


def display_cam(cam, cam_params, laser = None):
    """Continuous display of a camera (spatial : image + profile, spectral : image + integral) with the
    window of the stages (and of the lasers if laser is not None), the shutter opened if laser is not None."""
    if cam is cam_spat:
        mirror.set_position('spatial', verbose = True)
        x_win, y_win = cam_spat.display_below_profile()
    else:
        mirror.set_position('spectral', verbose = True)
        x_win, y_win = cam_spec.display_below_integral()
    if laser is not None:
        combi.control(x = x_win, y = y_win)   # control of the lasers
        y_win = y_win + combi.control_window_size()[1] + 20
        shutter.open()
    stage.stage_adjustment(obj_stage = obj_stage, cam = cam_spat if cam is cam_spat else None, x = x_win, y = y_win)
    time.sleep(1)
    play_one_pattern(DMD, DMD_initial_memory, cam_Par = cam_params, **pattern)
    if cam is cam_spat:
        cam_spat.display(display_max = True, display_profile = True)
    else:
        cam_spec.display(display_max = False, display_integral = True)
    DMD.Halt()
    shutter.close()
    mirror.set_position('spectral', verbose = True)
    print('stage : ' + str(stage.read_position()) + ', collection objective : ' + str(obj_stage.read_position()))
#%% ======================= PART A : HG-1 lamp (wavelength + resolution 'lamp') =======================
#%% A1 Prepare : lasers off, shutter closed. Fiber of the HG-1 at the position of the sample, lamp on since 5 min
combi.all_off()
shutter.close()
#%% A2 display spatial camera : center the fiber and focus with the collection objective (face of the fiber sharp)
# The light of the lamp does not pass through the DMD : the pattern is only played to trigger the camera.
display_cam(cam_spat, cam_spat_params)
#%% A3 display spectral camera : lines thin and bright (fine focus of the collection objective)
display_cam(cam_spec, cam_spec_params)
#%% A4 Acquisition of the lamp for each central wavelength
Lc_list    = np.arange(380, 921, 20)   # (nm) central wavelengths. Lamp lines : 365-579 (Hg) and 696-922 (Ar), few lines for Lc ~ 600-650
source     = 'HG-1_Oceanoptics'
data_name  = 'obj_fiber_source_' + source + '_spectral_calib_Gr_' + str(grating_nbr) + '_slit_' + str(round(slit_width)) + 'um_Lc_' + str(Lc_list[0]) + '-' + str(Lc_list[-1]) + 'nm'
data_path  = '../../data/' + data_folder_name + '/' + data_name
calib_path = data_path + '/spectral_calib_grating_' + str(grating_nbr)

combi.all_off()
shutter.close()
mirror.set_position('spectral', verbose = True)
measurements = []
expos_time = 0.05
for Lc in Lc_list:
    spectrograph.set_position(float(Lc))
    Lc_read = spectrograph.get_position()
    image, expos_time, saturated = snap_spec_auto(expos_time)
    spectrum, rows = sc.extract_spectrum(image)
    lines = sc.find_lines(spectrum, x, max_fwhm_px = spectrograph_params.slit_width / pixel_width)   # wide lines with a wide slit
    measurements.append({'Lc': Lc_read, 'expos_time': expos_time, 'x': x, 'spectrum': spectrum, 'image': image,
                         'lines': lines, 'saturated': saturated,
                         'sdk_lambda': spectrograph.get_calibration(sc.N_SENSOR, pixel_width)})
    print('Lc = ' + str(Lc_read) + ' nm, ti = ' + str(round(expos_time, 4)) + ' s, max = ' + str(image.max()) +
          ', rows ' + str(rows[0]) + '-' + str(rows[-1]) + ', ' + str(len(lines)) + ' peaks')
cam_spec_params_calib     = cam_Parameters(cam = cam_spec)                       # kept for the metadata (the check cell changes them)
spectrograph_params_calib = Spectrograph_Parameters(spectrograph = spectrograph)
#%% A5 Identification of the lines, fit of the model lambda(pixel, Lc) and of the resolution FWHM(pixel, Lc)
fit, table = sc.calibrate(measurements,
                          second_order  = True,   # 2nd order of the Hg lines (365, 405, 408, 436 nm -> 730, 809, 816, 872 nm) if there is no order sorting filter
                          deg_x         = 3,      # degree along the sensor
                          deg_Lc        = 2,      # degree along the central wavelength
                          max_shift     = 10,     # (nm) max error of the theoretical calibration of the SDK
                          tol           = 0.6,    # (nm) tolerance of the 1st identification
                          slit_width_um = spectrograph_params_calib.slit_width)   # the resolution is stored for this slit width
calib = sc.new_calib_entry(fit, grooves = spectrograph.get_grating().grooves,
                           slit_width_um = spectrograph_params_calib.slit_width,
                           data = calib_path + '.npz')
sc.plot_calibration(measurements, calib, table, save_path = data_path)
sc.save_calibration(calib_path + '.npz', measurements, calib, table)
print(calib)
#%% A6 Write the calibration in WAVELENGTH_CALIB (spas/spectral_calib_SPIM1D.py), check the figures before. Used immediately, no restart needed
# the mean resolution (380 - 930 nm) is stored for this slit width ; it becomes resolution_th if resolution_method = 'lamp'
sc.write_wavelength_calib(grating_nbr, calib, use_resolution = (resolution_method == 'lamp'))
spectrograph_params_calib.update_resolution(verbose = True)
print('resolution_th = ' + str(spectrograph_params_calib.resolution_th) + ' nm (slit ' + str(spectrograph_params_calib.slit_width) + ' µm)')
#%% A7 transfer data to girder : npz (images, spectra, lines), json (calibration) and figures, calibration in the metadata
transfer_data_spectral_calib(calib, spectrograph_params_calib, cam_spec_params_calib, measurements,
                             setup_version, data_folder_name, data_name, collection_access)
#%% A8 Check : spectrum of the lamp with the new calibration (any Lc, ROI and binning)
spectrograph_params = spectrograph.setup(grating_nbr = grating_nbr, position = 640, print_position = True)
cam_spec_params = cam_spec.setup(expos_time = 0.1, binningX = 8, binningY = 8, snapshot = True)
image, expos_time, saturated = snap_spec_auto(0.1)
plot_spectrum(image, cam_spec_params, spectrograph_params)
wavelengths = define_wavelengths_matrix(cam_spec_params, [(spectrograph_params.position, grating_nbr)], verbose = True)
resolution = spectral_resolution(cam_spec_params, [(spectrograph_params.position, grating_nbr)], verbose = True)   # (nm), binning 8 included
cam_spec_params = cam_spec.setup(expos_time = 0.05, binningX = 1, binningY = 8, snapshot = True)   # back to the calibration settings
#%% ======================= PART B : laser line (resolution 'laser', light sheet of the SPIM) =======================
#%% B1 Lasers. Fluorescent cuvette in place, notch filters removed (or laser not blocked by them)
lasers     = {473: {'power_mW': 0.5, 'linewidth_nm': 0},    # {wavelength (nm): power (mW), width of the laser line (nm, datasheet, 0 if << resolution)}
              594: {'power_mW': 0.5, 'linewidth_nm': 0}}
Lc_offsets = np.arange(-40, 41, 10)   # (nm) Lc = laser - offset : the line moves along the sensor (~ +/- 43 nm on the sensor)
combi.print_lasers()
combi.all_off()
first_laser = list(lasers)[0]
combi.set_power(first_laser, lasers[first_laser]['power_mW'], unit = 'mW')
combi.laser_on(first_laser)
spectrograph_params = spectrograph.setup(grating_nbr = grating_nbr, position = first_laser, print_position = True)
#%% B2 display spatial camera : light sheet in the cuvette at the position of the acquisitions, focus with the collection objective
display_cam(cam_spat, cam_spat_params, laser = first_laser)
#%% B3 display spectral camera : thinnest laser line (fine focus of the collection objective), no saturation (power in the laser window)
display_cam(cam_spec, cam_spec_params, laser = first_laser)
combi_params = combi.get_params(verbose = True)
#%% B4 Acquisition : scan of Lc around each laser
data_name_laser = ('obj_fluo-cuve_source_laser-' + '-'.join(f'{L:g}' for L in lasers) + 'nm_spectral_resolution_Gr_' +
                   str(grating_nbr) + '_slit_' + str(round(slit_width)) + 'um')
data_path_laser = '../../data/' + data_folder_name + '/' + data_name_laser

mirror.set_position('spectral', verbose = True)
measurements_laser = []
for L, laser in lasers.items():
    combi.all_off()
    combi.set_power(L, laser['power_mW'], unit = 'mW')
    combi.laser_on(L)
    expos_time = 0.01
    for offset in Lc_offsets:
        spectrograph.set_position(float(L - offset))
        Lc_read = spectrograph.get_position()
        shutter.open()
        try:
            image, expos_time, saturated = snap_spec_auto(expos_time)
        finally:
            shutter.close()
        spectrum, rows = sc.extract_spectrum(image)
        measurements_laser.append({'laser_nm': L, 'Lc': Lc_read, 'expos_time': expos_time, 'x': x, 'spectrum': spectrum,
                                   'image': image, 'saturated': saturated})
        print('laser ' + str(L) + ' nm, Lc = ' + str(Lc_read) + ' nm, ti = ' + str(round(expos_time, 4)) + ' s, max = ' +
              str(image.max()) + ', rows ' + str(rows[0]) + '-' + str(rows[-1]))
combi.all_off()
cam_spec_params_laser     = cam_Parameters(cam = cam_spec)
spectrograph_params_laser = Spectrograph_Parameters(spectrograph = spectrograph)
#%% B5 FWHM of the laser line, resolution (dispersion from WAVELENGTH_CALIB if PART A done, otherwise from the scan itself)
result, table_laser = sc.laser_resolution(measurements_laser, grating_nbr = grating_nbr,
                                          linewidths = {L: laser['linewidth_nm'] for L, laser in lasers.items()})
for L, laser in lasers.items():
    if float(L) in result['per_laser']:
        result['per_laser'][float(L)]['power_mW'] = laser['power_mW']
sc.plot_laser_resolution(measurements_laser, result, table_laser, save_path = data_path_laser)
sc.save_laser_resolution(data_path_laser + '/spectral_resolution_laser.npz', measurements_laser, result, table_laser)
#%% B6 Write the resolution in WAVELENGTH_CALIB for this slit width ; it becomes resolution_th if resolution_method = 'laser'
sc.write_resolution(grating_nbr, spectrograph_params_laser.slit_width, result, use_resolution = (resolution_method == 'laser'))
spectrograph_params_laser.update_resolution(verbose = True)
print('resolution_th = ' + str(spectrograph_params_laser.resolution_th) + ' nm (slit ' + str(spectrograph_params_laser.slit_width) + ' µm)')
#%% B7 transfer data to girder : npz (images, spectra), json (resolution) and figure, resolution in the metadata
transfer_data_spectral_calib(result, spectrograph_params_laser, cam_spec_params_laser, measurements_laser,
                             setup_version, data_folder_name, data_name_laser, collection_access)
#%% Disconnect
combi.all_off()
spectrograph.disconnect(goto_zero = False)
disconnect_DMD(DMD)
cam_spat.disconnect()
cam_spec.disconnect()
stage.disconnect(go_home = True)
shutter.disconnect()
mirror.disconnect()
obj_stage.disconnect(go_home = True)
combi.disconnect()
