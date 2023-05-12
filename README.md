# Single-Pixel Acquisition Software (SPAS)

SPAS is python package designed for single-pixel acquisition.

SPAS has been tested for controlling a [DLP7000](https://www.vialux.de/en/hi-speed-v-modules.html) Spatial Light Modulator and an [AvaSpec-ULS2048CL-EVO](https://www.avantes.com/products/spectrometers/starline/avaspec-uls2048cl-evo/) spectrometer. It should work as well for for similar equipment with a few changes.

SPAS is a companion package to the [SPyRiT](https://github.com/openspyrit/spyrit) package.


# Installation
The SPAS package can be installed on Linux, MacOs and Windows. However, it will be fully functional on Windows only due to DLL dependencies required for harware control.

We recommend using a virtual environment.

* Clone the SPAS repository

```powershell
git clone git@github.com:openspyrit/spas.git
```

Navigate to `./spas/` and install the SPAS package in editable mode

```powershell
pip install -r requirements.txt
pip install -e .
```

* Add DLLs (optional, for instrumentation control only)

    The following dynamic-link libraries (DLLs) were required to control our instrumentation

    * `avaspecx64.dll` provided by your Avantes distributor
    * `alpV42.dll` available [here](https://www.vialux.de/en/hi-speed-download.html) by installing the entire ALP4 library


* The DLLs should be placed inside the  `lib` folder. The typical directory structure is

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

# API Documentation
https://spas.readthedocs.io/

# Contributors (alphabetical order)
* Thomas Baudier
* Nicolas Ducros - [Website](https://www.creatis.insa-lyon.fr/~ducros/WebPage/index.html)
* Laurent Mahieu Williame

# How to cite?
When using SPAS in scientific publications, please cite the following paper:

* G. Beneti-Martin, L Mahieu-Williame, T Baudier, N Ducros, "OpenSpyrit: an Ecosystem for Reproducible Single-Pixel Hyperspectral Imaging," Optics Express, Vol. 31, No. 10, (2023). https://doi.org/10.1364/OE.483937.

# License
This project is licensed under the LGPL-3.0 license - see the [LICENSE.md](LICENSE.md) file for details

# Getting started
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

* Measurements are saved on the disk (fully sampled)

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

* We consider an existing acquisition that was saved on the disk in the `../meas/` folder 

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

* We consider that we have access to a trained network and the covariance matrix associated to it. 

An example network can be downloaded [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/622b5ea843258e76eab21740). It allows the reconstruction of a 128 x 128 image from only 4096 Hadamard coefficients (i.e., 8192 raw measurements) that correspond to a full acquisition at a 64 x 64  resolution. Its associated covariance matrix can be downloaded [here](https://pilot-warehouse.creatis.insa-lyon.fr/#collection/6140ba6929e3fc10d47dbe3e/folder/63d7f3620386da2747641e1b).

``` python
from spas import ReconstructionParameters, setup_reconstruction

network_param = ReconstructionParameters(
    # Reconstruction network    
    M = 64*64,          # Number of measurements
    img_size = 128,     # Image size
    arch = 'dc-net',    # Main architecture
    denoi = 'unet',     # Image domain denoiser
    subs = 'rect',      # Subsampling scheme
    
    # Training
    data = 'imagenet',  # Training database
    N0 = 10,            # Intensity (max of ph./pixel)
    
    # Optimisation (from train2.py)
    num_epochs = 30,       # Number of training epochs
    learning_rate = 0.001, # Learning Rate
    step_size = 10,        # Scheduler Step Size
    gamma = 0.5,           # Scheduler Decrease Rate   
    batch_size = 256,      # Size of the training batch
    regularization = 1e-7 # Regularisation Parameter
    )
        
cov_path = '../stat/Cov_8_128x128.npy'
model_folder = '../model/'
      
model, device = setup_reconstruction(cov_path, model_folder, network_param)
```

Load noise calibration parameters (provided with the data or computed using tools in `/noise-calibration`). :warning: Noise parameters are not used anymore in the current implementation of `spas`.
 
``` python
from spas import load_noise
noise = load_noise('../noise-calibration/fit_model.npz')
```

Bin the spectral measurements (here, 4 bins between 530 nm and 730 nm)

``` python
from spas import spectral_binning
meas_bin, wavelengths_bin, _ = spectral_binning(meas.T, wavelengths, 530, 730, 4)
```

Reorder and subsample the spectral measurements
``` python
from spas.reconstruction_nn import reorder_subsample
meas_bin_2 = reorder_subsample(meas_bin, acquisition_param, network_param) 
```

Reconstruct the spectral images
``` python
from spas import reconstruct
rec = reconstruct(model, device, meas_bin_2)
```

Plot the spectral images
``` python
from spas import plot_color           
plot_color(rec, wavelengths_bin)
```
