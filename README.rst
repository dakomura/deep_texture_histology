***************************************************************************************
deep_texture_histology : Tools for deep texture representation for histology images.
***************************************************************************************

.. image:: https://github.com/dakomura/deep_texture_histology/blob/main/docs/_static/logo/dtr_logo.jpg

Overview
==============
deep_texture_representation is a python library to calculate deep texture representations (DTRs) for histology images (Cell Reports, 2022).
Fucntions for plotting the distribution of DTRs and content-based image retrieval are also implemented.

Installation
=========================
The package can be installed with ``pip``:

.. code:: console

   $ pip install deeptexture


Prerequisites
==============

Python version 3.6 or newer.

* numpy >=1.20.3
* pytorch >=1.7.1
* joblib >=0.13.2
* Pillow >=8.0.1
* nmslib >=2.0.6
* matplotlib >= 3.5.0
* scikit-learn >=1.1.0
* seaborn >=0.10.1
* pandas >=1.1.0
* cv2

Recommended Environment
=======================

* OS
   * Linux
   * Mac

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
