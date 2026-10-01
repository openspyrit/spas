from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='spas',
    version='1.4.0',
    include_package_data=True,
    description='A python toolbox for acquisition of images based on the single-pixel framework.',
    author='Guilherme Beneti Martins',
    url='https://github.com/openspyrit/spas',
    long_description=readme,
    long_description_content_type = "text/markdown",
    install_requires=[
        'ALP4lib @ git+https://github.com/openspyrit/ALP4lib@3db7bec88b260e5396626b1b185d7a2f678e9bbf',
        'dataclasses-json (==0.5.2)',
        'certifi',
        'cycler',
        'kiwisolver',
        'matplotlib', #==3.7.5
        'numpy',
        # 'msl-equipment @ git+https://github.com/MSLNZ/msl-equipment.git',
        'Pillow',
        'pyparsing',
        'python-dateutil',
        'six',
        'tqdm (==4.60.0)',
        'torch',
        'torchvision',
        'spyrit',
        'wincertstore',
        'pyueye',
        'tensorboard',
        'girder-client',
        'imageio',
        'opencv-python',
        'pylablib',
        'progress',
        'PIPython',
        'pyserial',
        'pythonnet',
        'ipython',
        'plotter',
        'tikzplotlib',
        # Official Andor packages for the SPIM (spectro_Shamrock_module.py and cam_Andor_module.py).
        # They are not on PyPI, they are provided with the Andor SDKs (installed with Solis) and must be installed before spas:
        #   pyAndorSpectrograph : C:/Program Files/Andor SDK/Python/pyAndorSpectrograph   (SDK2, Shamrock spectrograph)
        #   pyAndorSDK3         : C:/Program Files/Andor SDK3/Python/pyAndorSDK3          (SDK3, Zyla cameras)
        # Their setup.py write in their own folder, which is read-only in Program Files: copy the folder before installing, e.g.
        #   xcopy /E /I "C:\Program Files\Andor SDK\Python\pyAndorSpectrograph" %TEMP%\pyAndorSpectrograph
        #   pip install %TEMP%\pyAndorSpectrograph
        #   xcopy /E /I "C:\Program Files\Andor SDK3\Python\pyAndorSDK3" %TEMP%\pyAndorSDK3
        #   pip install %TEMP%\pyAndorSDK3
        'pyAndorSpectrograph',
        'pyAndorSDK3'
    ],
    packages=find_packages()
)
