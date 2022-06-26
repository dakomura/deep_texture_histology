Quick Start
===========

This tutorial introduces you to the basics of deep_texture_histology workflow.

DTR calculation
---------------

.. code-block:: python

    import deeptexture as dt

    # create DTR object
    dtr_obj = dt.DTR()

    # calculate DTR for one image
    img = "./1.jpg"
    dtr = dtr_obj.get_dtr(img)

Plot DTR distribution
---------------------

1.  Plot images

    .. code-block:: python

        import deeptexture as dt

        # create DTR object
        dtr_obj = dt.DTR()

        # calculate DTR for one image
        img = "./1.jpg"
        dtr = dtr_obj.get_dtr(img)

Content-based image retrieval
-----------------------------

CBIR is.


1.  Create CBIR database.

    .. code-block:: python

        import deeptexture as dt

        # create DTR object
        dtr_obj = dt.DTR()

        # calculate DTR for one image
        img = "./1.jpg"
        dtr = dtr_obj.get_dtr(img)
