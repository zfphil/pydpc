from setuptools import setup, find_packages
import os, sys
import subprocess

# Define version
__version__ = 0.02

setup( name             = 'pydpc'
     , version          = __version__
     , description      = 'Python implementation of differential phase contrast'
     , license          = 'BSD'
     , packages         = find_packages()
     , include_package_data = True
     , install_requires = ['planar', 'sympy', 'numexpr', 'contexttimer', 'imageio', 'matplotlib_scalebar', 'tifffile', 'pyserial', 'numpy', 'scipy', 'scikit-image', 'planar']
     )
