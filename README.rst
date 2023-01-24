***************************************************************************************
deep_texture_histology : Tools for deep texture representation for histology images.
***************************************************************************************

Overview
==============
deep_texture_representation is a python library to calculate deep texture representations (DTRs) for histology images (Cell Reports, 2022).
Fucntions for plotting the distribution of DTRs, content-based image retrieval, and supervised learning are also implemented.

Installation
=========================
The package can be installed with ``pip``:

.. code:: console

   $ pip install deeptexture

Conda environmental files including dependent libraries for various OS are available `here <https://github.com/dakomura/dtr_env>`_.

To test the successful installation,

.. code-block:: console

   $ git clone https://github.com/dakomura/deep_texture_histology
   $ cd deep_texture_histology
   $ python check_libraries_and_quick_test.py

Prerequisites
==============

Python version 3.6 or newer.

* numpy
* tensorflow
* joblib
* Pillow
* nmslib
* matplotlib
* scikit-learn
* seaborn
* pandas
* cv2

All the required libraries can be installed with conda yml files.
See https://github.com/dakomura/dtr_env

Recommended Environment
=======================

* OS
   * Linux (both CPU and GPU version)
   * Mac (both CPU and GPU version for M1 and M2 chip)
   * Windows (both CPU and GPU version)

License
=======

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA 4.0)

For non-commercial use, please use the code under CC-BY-NC-SA.

If you would like to use the code for commercial purposes, please contact us <ishum-prm@m.u-tokyo.ac.jp>.

Citation
========

If you use this library for your research, please cite:

    Komura, D., Kawabe, A., Fukuta, K., Sano, K., Umezaki, T., Koda, H., Suzuki, R., Tominaga, K., Ochi, M., Konishi, H., Masakado, F., Saito, N., Sato, Y., Onoyama, T., Nishida, S., Furuya, G., Katoh, H., Yamashita, H., Kakimi, K., Seto, Y., Ushiku, T., Fukayama, M., Ishikawa, S., 
    
    "*Universal encoding of pan-cancer histology by deep texture representations.*"
    
    Cell Reports 38, 110424,2022. https://doi.org/10.1016/j.celrep.2022.110424

Documentation
=============

`Documentation <https://deep-texture-histology.readthedocs.io/en/latest/>`_
