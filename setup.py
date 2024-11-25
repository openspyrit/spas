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
        'matplotlib==3.7.5',
        'numpy',
        'msl-equipment @ git+https://github.com/MSLNZ/msl-equipment.git',
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
        'plotter',
        'tikzplotlib'
    ],
    packages=find_packages()
)
