# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:44:46 2026

@author: mahieu
"""

#%% test : minimum period between two triggers (spectral camera)
from spas.acquisition_SPIM1D import check_timestamps


mirror.set_position('spectral', verbose = True)
shutter.open()
for add_ill in [26000, 28000, 30000, 33000, 36000, 40000, 45000, 50000]:
    DMD_params = setup_DMD(DMD = DMD, DMD_initial_memory = DMD_initial_memory, acquisition_params = acquisition_params,
                           integration_time = ti, add_illumination_time = add_ill)
    cam_spec.setup_acquisition(nframes = acquisition_params.pattern_amount)
    cam_spec.start_acquisition()
    ts = []
    DMD.Run(loop = False)
    try:
        while len(ts) < acquisition_params.pattern_amount:
            cam_spec.read_frame(timeout = 2)
            ts.append(cam_spec.last_timestamp)
    except RuntimeError:
        pass
    DMD.Halt()
    cam_spec.stop_acquisition()
    p = np.diff(ts) * 1e3
    print(f'### add_illumination = {add_ill/1000:5.1f} ms, picture time = {DMD_params.picture_time_us/1000:7.3f} ms : '
          f'{len(ts)}/{acquisition_params.pattern_amount} frames, period median = {np.median(p):.3f} ms, max = {np.max(p):.3f} ms')
    check_timestamps(np.array(ts), 'spectral')
shutter.close()