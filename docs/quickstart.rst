Quick Start
===========

This tutorial introduces you to the basics of deep_texture_histology workflow.

DTR calculation
---------------

.. code-block:: python

    import deeptexture as dt

    # create DTR object
    dtr_obj = dt.DTR(arch='vgg', layer='block3_conv3', dim=1024)

    # calculate DTR for one image file
    imgfile = './1.jpg'
    dtr = dtr_obj.get_dtr(imgfile)
    
    # calculate DTR for one rotated image file
    imgfile = './1.jpg'
    dtr = dtr_obj.get_dtr(imgfile)
    dtr_rot = dtr_obj.get_dtr(imgfile, angle=90)
    
    # calculate DTR for one image object
    imgfile = './1.jpg'
    dtr = dtr_obj.get_dtr(imgfile)
    img = Image.open(imgfile)
    dtr = dtr_obj.get_dtr(img)

    # calculate DTR for one image object
    imgfile = './1.jpg'
    dtr = dtr_obj.get_dtr(imgfile)
    img_np = np.array(Image.open(imgfile))
    dtr = dtr_obj.get_dtr(img_np)

    # calculate DTRs for multiple images
    imgfiles = glob.glob("./*.jpg")
    dtrs = dtr_obj.get_dtr_multifiles(imgfiles)

    # calculate DTRs for multiple rotated images
    imgfiles = glob.glob("./*.jpg")
    dtrs = dtr_obj.get_dtr_multifiles(imgfiles, angle=90)

Plot DTR distribution
---------------------

1.  Plot images

    .. code-block:: python

        from deeptexture import plt_dtr

        imgfiles = ['01.jpg',
                    '02.jpg',
                    '03.jpg',
                    '04.jpg',]
        text = ['01','02','03','04',]

        dtrs = dtr_obj.get_dtr_multifiles(imgfiles, angle=90)

        #Principal component analysis is applied to DTRs and PC1 and PC2 is shown.
        X_emb = plt_dtr.plt_dtr_image(dtrs, imgfiles, method="pca", x1=1, x2=2, text=text)
        
        #Calculated embedding can be the input.
        _ = plt_dtr.plt_dtr_image(X_emb, imgfiles, text=text)

2.  Plot attributes

    .. code-block:: python

        from deeptexture import plt_dtr

        imgfiles = ['01.jpg',
                    '02.jpg',
                    '03.jpg',
                    '04.jpg',]
        attr =  ['cancer',
                'cancer',
                'normal',
                'normal',]
        text = ['01','02','03','04',]

        dtrs = dtr_obj.get_dtr_multifiles(imgfiles, angle=90)

        #Principal component analysis is applied to DTRs and PC1 and PC2 is shown.
        df = plt_dtr.plt_dtr_attr(dtrs, attr, method="pca", x1=1, x2=2, text=text)

        #Calculated embedding can be the input.
        _ = plt_dtr.plt_dtr_attr(X_emb, attr, text=text)


Content-based image retrieval
-----------------------------

DTR enables quick and accurate retrieval of histologically similar images using CBIR module.
You can create original database and save the files in the specified directory.


1.  Create CBIR database.

    .. code-block:: python

        import pandas as pd

        import deeptexture as dt
        from deeptexture import cbir

        # create DTR object
        dtr_obj = dt.DTR(arch='vgg', layer='block3_conv3', dim=1024)

        # create CBIR object
        cbir_obj = cbir.CBIR(dtr_obj, project='DB', working_dir='CBIR')

        # create CBIR database
        imgfiles = ['01.jpg',
                    '02.jpg',
                    '03.jpg',
                    '04.jpg',]
        patients = ['01',
                    '02',
                    '03',
                    '04',]
        attr =  ['cancer',
                'cancer',
                'normal',
                'normal',]
        df_attr = pd.DataFrame({'imgfile': imgfiles,
                                'patient': patients,
                                'tissue',: attr)
        
        cbir_obj.create_db(df_attr, img_attr='imgfile', save=True)

2.  Search similar histology images.

    .. code-block:: python

        # search the most similar images (top two)
        qimgfile = "./5.jpg"
        cbir_obj.search(qimgfile, img_attr='imgfile', case_attr='patient', n=2)
