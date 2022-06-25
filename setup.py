# Author: Daisuke Komura <kdais-prm@m.u-tokyo.ac.jp>
# Copyright (c) 2022 Daisuke Komura
# License: This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA 4.0)

from setuptools import setup
import dtr

DESCRIPTION = "histo_dtr: Deep Texture Representations for Cancer Histology Images"
NAME = 'histo-dtr'
AUTHOR = 'Daisuke Komura'
AUTHOR_EMAIL = 'kdais-prm@m.u-tokyo.ac.jp'
URL = 'https://github.com/dakomura/deep_texture_histology'
LICENSE = 'CC-BY-NC-SA 4.0'
DOWNLOAD_URL = 'https://github.com/dakomura/deep_texture_histology'
VERSION = dtr.__version__
PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    'tifffile >=2022.5.4',
    'numpy >=1.20.3',
    'tripy >=1.0.0',
    'pyclipper >=1.3.0',
    'opencv-python >= 4.6.0',
    'zarr >=2.11.3',
    'imagecodecs >=2022.2.22',
    'click >=8.1.3',
]

PACKAGES = [
    'histo_dtr'
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Multimedia :: Graphics',
]

with open('README.rst', 'r', encoding='utf-8') as fp:
    long_description = fp.read()

setup(name=NAME,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=AUTHOR,
      maintainer_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      long_description=long_description,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      packages=PACKAGES,
      classifiers=CLASSIFIERS,
    )