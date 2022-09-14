from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='spas',
    version='0.0.1',
    description='A python toolbox for acquisition of images based on the single-pixel framework.',
    author='Guilherme Beneti Martins',
    url='https://github.com/openspyrit/spas',
    long_description=readme,
    long_description_content_type = "text/markdown",
    install_requires=[
        'ALP4lib @ git+https://github.com/guilhermebene/ALP4lib.git@7e35abf3a5c2e31652f7cfb2e4243b279b6a3a47',
        'dataclasses-json (==0.5.2)',
        'certifi',
        'cycler',
        'kiwisolver',
        'matplotlib',
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
        'girder-client'
    ],
    packages=find_packages()
)
