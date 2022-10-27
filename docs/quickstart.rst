Quick Start
===========

This tutorial introduces you to the basics of deep_texture_histology workflow.

DTR calculation
---------------

.. code-block:: python

    import deeptexture as dt
    import glob

    # create DTR object
    dtr_obj = dt.DTR(arch='vgg', layer='block4_conv3', dim=1024)

    # calculate DTR for one image file
    imgfile = './example/0KNpXTsaNix2T6.jpg'
    dtr = dtr_obj.get_dtr(imgfile)
    
    # calculate mean DTR for unrotated and rotated image file
    imgfile = './example/0KNpXTsaNix2T6.jpg'
    dtr = dtr_obj.get_dtr(imgfile)
    dtr_rot = dtr_obj.get_dtr(imgfile, angle=[0, 90])
    
    # calculate DTR for one image object
    imgfile = './example/0KNpXTsaNix2T6.jpg'
    dtr = dtr_obj.get_dtr(imgfile)
    img = Image.open(imgfile)
    dtr = dtr_obj.get_dtr(img)

    # calculate DTR for one image object
    imgfile = './example/0KNpXTsaNix2T6.jpg'
    dtr = dtr_obj.get_dtr(imgfile)
    img_np = np.array(Image.open(imgfile))
    dtr = dtr_obj.get_dtr(img_np)

    # calculate DTRs for multiple images
    imgfiles = glob.glob("./example/*.jpg")
    dtrs = dtr_obj.get_dtr_multifiles(imgfiles)

    # calculate DTRs for multiple unrotated and rotated images
    imgfiles = glob.glob("./example/*.jpg")
    dtrs = dtr_obj.get_dtr_multifiles(imgfiles, angle=[0, 90])

Plot DTR distribution
---------------------

1.  Plot images

    .. code-block:: python

        from deeptexture import plt_dtr

        imgfiles = ['./example/0KNpXTsaNix2T6.jpg',
                    './example/0zpw5BpfoiRX9m.jpg',
                    './example/11303cFBFubBwG.jpg',
                    './example/19HDnAR7OBcA1d.jpg',]
        text = ['01','02','03','04',]

        dtrs = dtr_obj.get_dtr_multifiles(imgfiles, angle=[0, 90])

        #Principal component analysis is applied to DTRs and PC1 and PC2 is shown.
        X_emb = plt_dtr.plt_dtr_image(dtrs, imgfiles, method="pca", x1=1, x2=2, text=text)
        
        #Calculated embedding can be the input.
        _ = plt_dtr.plt_dtr_image(X_emb, imgfiles, text=text)

2.  Plot attributes

    .. code-block:: python

        from deeptexture import plt_dtr

        imgfiles = ['./example/0KNpXTsaNix2T6.jpg',
                    './example/0zpw5BpfoiRX9m.jpg',
                    './example/11303cFBFubBwG.jpg',
                    './example/19HDnAR7OBcA1d.jpg',]
        attr =  ['cancer',
                'cancer',
                'normal',
                'normal',]
        text = ['01','02','03','04',]

        dtrs = dtr_obj.get_dtr_multifiles(imgfiles, angle=[0, 90])

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
        dtr_obj = dt.DTR(arch='vgg', layer='block4_conv3', dim=1024)

        # create CBIR object
        cbir_obj = cbir.CBIR(dtr_obj, project='DB', working_dir='CBIR')

        # create CBIR database
        imgfiles = ['./example/0KNpXTsaNix2T6.jpg',
                    './example/0zpw5BpfoiRX9m.jpg',
                    './example/11303cFBFubBwG.jpg',
                    './example/19HDnAR7OBcA1d.jpg',]
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
                                'tissue': attr})
        
        cbir_obj.create_db(df_attr, img_attr='imgfile', save=True)

2.  Search similar histology images.

    .. code-block:: python

        # search the most similar images (top two)
        qimgfile = "./example/3t0U7aBXRY9v1V.jpg"
        cbir_obj.search(qimgfile, img_attr='imgfile', case_attr='patient', n=2)

Supervised learning model
-------------------------

.. code-block:: python

    import deeptexture as dt
    from deeptexture import ml
    
    imgfiles = ['./example/0KNpXTsaNix2T6.jpg',
                './example/0zpw5BpfoiRX9m.jpg',
                './example/11303cFBFubBwG.jpg',
                './example/19HDnAR7OBcA1d.jpg',]
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
                            'tissue': attr})

    dtrs = dtr_obj.get_dtr_multifiles(imgfiles, angle=[0, 90]) 
    
    # create ml object
    ml_obj = ml.ML(dtrs, df_attr.imgfile)
    
    # create a model to classify image into tissue type
    y = df_attr.tissue
    cases = df_attr.patient
    
    lr = ml_obj.fit_eval(y, cases)
