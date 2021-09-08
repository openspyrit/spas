from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

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
    install_requires=required
    packages=find_packages())
