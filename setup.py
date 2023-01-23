#!/usr/bin/env python
# %Part of https://github.com/moritzhuetten/DMbounds under the 
# %Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License, see LICENSE.rst 
# import sys
from setuptools import setup, find_packages

setup(
    name='dmbounds',
    version='0.1.1',    
    description='A library of current DM bounds',
    url='https://github.com/moritzhuetten/dmbounds',
    author='Moritz Huetten, Michele Doro',
    author_email='huetten@icrr.u-tokyo.ac.jp',
    license='CC Attribution-NonCommercial-ShareAlike 3.0 Unported',
    packages=find_packages(),
    install_requires=[
        'palettable',
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
        'pandas',
        'ipywidgets',
        'IPython',
        'PyYAML'
    ],
    package_data={
        'dmbounds': [
            'bounds/*/*',
            'modelpredictions/*',
            'legends/*'
        ],
    },
)
