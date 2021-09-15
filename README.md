# Single-Pixel Acquisition Software (SPAS) - Python version

A python toolbox for acquisition of images based on the single-pixel framework.

## Installation (Windows only)

1.  Create a new environment (tested under conda)

```powershell
conda create --name my_spas_env
conda activate my_spas_env
conda install -c anaconda pip 
```

2. Install the [SPyRiT](https://github.com/openspyrit/spyrit) package (tested with version 1.0.0). Typically

```powershell
pip install requests torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install spyrit==1.0.0
```

2. Clone the SPAS repository

```powershell
git clone git@github.com:openspyrit/spas.git
```

Navigate to `./spas/` and install the SPAS package in editable mode

```powershell
pip install -r requirements.txt
pip install -e .
```

3. Add DLLs 

The following dynamic-link libraries (DLLs) are required

* `avaspecx64.dll` provided by your Avantes distributor
* `alpV42.dll` available [here](https://www.vialux.de/en/hi-speed-download.html) by installing the entire ALP4 library

They should be placed inside the  `lib` folder

4. The typical directory structure is

```
├───lib
│   ├───alpV42
│   │   └───x64
│   │   │   └───alpV42.dll
│   └───avaspec3
│   │   └───avaspecx64.dll
├───noise-calibration
├───models
├───scripts
├───spas
├───requirements.txt
├───setup.py
├───stats
│   ├───Average_64x64.npy
│   ├───Cov_64x64.npy
```

## Preparation (just once)
### 1. Creating Walsh-Hadamard patterns

Run in Python:
``` python
from spas import walsh_patterns
walsh_patterns(save_data=True)
```
By default the patterns are 1024x768 PNG images saved in `./Walsh_64_64/`.

### 2. Get statistics

   * Download covariance and mean matrices and save them both in `./stats/`:
  
        https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_ISTE/Average_64x64.npy

        https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_ISTE/Cov_64x64.npy


   * (Alternative, which takes much longer) compute covariance matrix.
``` python 
from spyrit.misc.statistics import stat_walsh_stl10
stat_walsh_stl10()
```

### 3. Generate the pattern order
It is necessary to generate an array that specifies the order in which the patterns are acquired. This can be done by running:

``` python
from spas import generate_hadamard_order
generate_hadamard_order(N=64, name='pattern_order', cov_path='./stats/Cov_64x64.npy', pos_neg=True)
```

The output array will be placed in `./stats/pattern_order.npz`. It may be necessary to change `cov_path` depending on the previous step.

### 4. We recommend using spyder for acquisition

In a terminal :
``` powershell
pip install spyder
spyder
```

You may experience an issue launcing spyder (July 2021). It is solved with:
``` powershell
conda install pywin32
```

  ## First Acquisition

We provides several script examples in `./scripts/`. A minimal working example is provided below.

* Initialization (just once, two consecutively returns an error):
``` python 
from spas import *
spectrometer, DMD, DMD_initial_memory = init() 
```

* Setup:
``` python   
metadata = MetaData(
    output_directory='./meas/',
    pattern_order_source='./stats_download/pattern_order.npz',
    pattern_source='./Walsh_64x64/',
    pattern_prefix='Walsh_64x64',
    experiment_name='my_first_measurement',
    light_source='white_lamp',
    object='no_object',
    filter='no_filter',
    description='my_first_description')
    
acquisition_parameters = AcquisitionParameters(
    pattern_compression=1.0,
    pattern_dimension_x=64,
    pattern_dimension_y=64)
    
spectrometer_params, DMD_params = setup(
    spectrometer=spectrometer, 
    DMD=DMD,
    DMD_initial_memory=DMD_initial_memory,
    metadata=metadata, 
    acquisition_params=acquisition_parameters,
    integration_time=1.0)

```

* Acquisition:
``` python 
meas = acquire(
    ava=spectrometer,
    DMD=DMD,
    metadata=metadata,
    spectrometer_params=spectrometer_params,
    DMD_params=DMD_params,
    acquisition_params=acquisition_parameters,
    repetitions=1,
    reconstruct=False)

```

* Disconnect, otherwise it is possible to run setup and/or acquisition again.
``` python 
disconnect(spectrometer, DMD)
```

## First Reconstruction (No Neural Networks)

Here, we consider an acquisition with `pattern_compression=1.0`, meaning all the patterns are acquired.

* Measurements are in memory (fully sampled)
  

Reconstruct the measurements contained in variable `meas`:
``` python 
import spyrit.misc.walsh_hadamard as wh
from spas import reconstruction_hadamard
H = wh.walsh2_matrix(64)
rec = reconstruction_hadamard(acquisition_parameters.patterns, 'walsh', H, meas)
```

Bin the reconstructed hypercube in 8 bins between 530 and 730 nm:

``` python
from spas import spectral_binning
rec_bin, wavelengths_bin, _ = spectral_binning(rec.T, acquisition_parameters.wavelengths, 530, 730, 8)
```

Plot the 8 spectral bins:
``` python
from spas import plot_color
plot_color(rec_bin, wavelengths_bin)
```

* Measurements are saved in the disk (fully sampled)

Reconstruct the measurements saved as `../meas/my_first_measurement`.

Read the data from a file:
``` python
import numpy as np
file = np.load('../meas/my_first_measurement' + '_spectraldata.npz')
meas = file['spectral_data']
```

Read the metadata (it is necessary to recover the acquisition order of the patterns from `acquisition_parameters`):
``` python
from spas import read_metadata, reconstruction_hadamard
_, acquisition_parameters, _, _ = read_metadata('../meas/my_first_measurement' + '_metadata.json')
```

Reconstruct:
``` python
import spyrit.misc.walsh_hadamard as wh
H = wh.walsh2_matrix(64)
rec = reconstruction_hadamard(acquisition_metadata.patterns, 'walsh', H, meas)
```
## Reconstruction with a Neural Network

* Measurements are on the disk (fully-sampled here, works too with `pattern_compression=.25`) 

Read the data:
``` python
import numpy as np
file = np.load('../meas/my_first_measurement' + '_spectraldata.npz')
meas = file['spectral_data']
```

Read the metadata (we need the get the acquisition order of the patterns):
``` python
from spas import read_metadata, reconstruction_hadamard
_, acquisition_parameters, _, _ = read_metadata('../meas/my_first_measurement' + '_metadata.json')
```

* We have access to a trained network:

An example network can be downloaded [here](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2021_ISTE/NET_c0mp_N0_50.0_sig_0.0_Denoi_N_64_M_1024_epo_40_lr_0.001_sss_20_sdr_0.2_bs_256_reg_1e-07.pth) , which you can save in `./models/`. It allows to reconstruction images from only 1024 hadamard coefficients (i.e., 2048 raw measurements):

``` python
from spas import ReconstructionParameters, setup_reconstruction
network_params = ReconstructionParameters(
    img_size=64,
    CR=1024,
    denoise=True,
    epochs=40,
    learning_rate=1e-3,
    step_size=20,
    gamma=0.2,
    batch_size=256,
    regularization=1e-7,
    N0=50.0,
    sig=0.0,
    arch_name='c0mp',)
        
cov_path = '../stats/Cov_64x64.npy'
mean_path = '../stats/Average_64x64.npy'
model_root = '../models/'

import spyrit.misc.walsh_hadamard as wh
H = wh.walsh2_matrix(64)/64        
model, device = setup_reconstruction(cov_path, mean_path, H, model_root, network_params)
```

Load noise calibration parameters (provided with the data or computed using tools in `/noise-calibration`)
``` python
from spas import load_noise
noise = load_noise('../noise-calibration/fit_model.npz')
```

Bin before reconstruction and plot

``` python
from spas import spectral_binning
meas_bin, wavelengths_bin, _, noise_bin = spectral_binning(meas.T, acquisition_parameters.wavelengths, 530, 730, 8, noise)
```

Reconstruction and plot
``` python
from spas import reconstruct, plot_color
rec = reconstruct(model, device, meas_bin[0:8192//4,:], 1, noise_bin)           
plot_color(rec, wavelengths_bin)
```