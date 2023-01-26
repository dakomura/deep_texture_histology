.. highlight:: shell

============
Installation
============

deeptexture can be installed via pip.
The installation command is::

    $ pip install deeptexture 

But we recommend installing deeptexture and all required packages in a dedicated Anaconda environment as follows::

    $ conda env create -f deeptexture_XXX.yml

The conda environmental files including dependent libraries for various OS are available `here <https://github.com/dakomura/dtr_env>`_.

=================
Installation test
=================

The installation can be considered successful if no error is generated when this script is run.

.. code-block:: bash

    $ conda activate deeptexture 
    $ git clone https://github.com/dakomura/deep_texture_histology
    $ cd deep_texture_histology
    $ python check_libraries_and_quick_test.py
