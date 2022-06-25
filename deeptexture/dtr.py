from typing import Any, List, Union
from joblib import Parallel, delayed
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, preprocessing
from tensorflow.keras.applications import resnet50, vgg16, mobilenet_v2, inception_v3, nasnet, densenet, inception_resnet_v2

#efficientnet is optional
import importlib
eff_spec = importlib.util.find_spec("efficientnet")
if eff_spec is not None:
    import efficientnet.tfkeras


class DTR:
    def __init__(self,
                 arch: str = 'vgg',
                 layer: str = 'block3_conv3', 
                 dim: int = 1024, 
                 ) -> None:
        """Initializes DTR model and its preprocessing function.

        Args:
            arch (str, optional): CNN model. Defaults to 'vgg'.
            layer (str, optional): A layer in the CNN model. Defaults to 'block3_conv3'.
            dim (int, optional): The output dimension. Defaults to 1024.
        """
        self.arch = arch
        self.layer = layer
        self.dim = dim

        self.archs_dict = {
            'mobilenet': mobilenet_v2.MobileNetV2,
            'vgg': vgg16.VGG16,
            'resnet50': resnet50.ResNet50,
            'inceptionv3': inception_v3.InceptionV3,
            'nasnet': nasnet.NASNetLarge,
            'densenet': densenet.DenseNet201,
            'inceptionresnetv2': inception_resnet_v2.InceptionResNetV2,
        }
        self.prep_dict = {
            'mobilenet': mobilenet_v2,
            'vgg': vgg16,
            'resnet50': resnet50,
            'inceptionv3': inception_v3,
            'nasnet': nasnet,
            'densenet': densenet,
            'inceptionresnetv2': inception_resnet_v2,
        }

        if eff_spec is not None:
           self.archs_dict['efficientnet'] = efficientnet.tfkeras.EfficientNetB7
           self.prep_dict['efficientnet'] = efficientnet.tfkeras 

        self._create_model(self.archs_dict[self.arch], 
                                    self.layer, 
                                    self.dim,
                                    )

        # preprocess function
        self.prep = self.prep_dict[arch].preprocess_input


    def _create_model(self,
                     arch: str,
                     layer_name: str,
                     ) -> None:

        conv_base = arch(
            weights = "imagenet",
            include_top = None,
            input_shape = (None, None, 3))

        x1 = conv_base.get_layer(layer_name).output
        _,_,_,c = x1.shape

        rng = np.random.default_rng(2022)

        r1 = rng.uniform(0,1,(1,c,self.dim))
        nfilter1 = np.where(r1>0.5,1,-1)
        filter1 = tf.constant(nfilter1,dtype=float)

        r2 = rng.uniform(0,1,(1,c,self.dim))
        nfilter2 = np.where(r2>0.5,1,-1)
        filter2 = tf.constant(nfilter2,dtype=float)

        x2 = tf.keras.layers.Reshape((-1,c))(x1)

        y1 = tf.nn.conv1d(x2,filter1,stride=1,padding='SAME',data_format='NWC')
        y2 = tf.nn.conv1d(x2,filter2,stride=1,padding='SAME',data_format='NWC')

        y3 = tf.keras.layers.Reshape((-1,self.dim))(y1)
        y4 = tf.keras.layers.Reshape((-1,self.dim))(y2)

        z1 = tf.keras.layers.Multiply()([y3,y4])
        z2 = tf.keras.layers.GlobalAveragePooling1D()(z1)
        z3 = tf.math.sqrt(tf.math.abs(z2))*tf.math.sign(z2)
        z4 = tf.math.l2_normalize(z3)

        self.cbp = models.Model(conv_base.input,z4)

    def _process_image(self, 
                       img_path: str, 
                       angle: Union[None, int]
                       ) -> np.ndarray:

        img = preprocessing.image.load_img(img_path)
        x = preprocessing.image.img_to_array(img)
        if angle is not None:
            x = preprocessing.image.apply_affine_transform(x, theta = angle)
        x = np.expand_dims(x, axis=0)
        x = self.prep(x)
        return (x)
    
    def get_dtr(self, 
                img: Any, 
                angle: Union[None, int] = None
                ) -> np.ndarray:
        """Calculates DTR for an image object or file.

        Args:
            img (Any): Image file or image object (numpy array or PIL Image object)
            angle (Union[None, int], optional): Rotation angle (0-360). Defaults to None.

        Returns:
            np.ndarray: DTR for the image
        """
        if type(img) == str:
            img = Image.open(img).convert("RGB")
        elif not type(img) == np.ndarray:
            x = np.array(img)

        if angle is not None:
            x = preprocessing.image.apply_affine_transform(x, theta = angle)

        x = np.expand_dims(x, axis=0)
        x = self.prep(x)
        return self.cbp([x])[0]

    def sim(self, 
            x: np.ndarray,
            y: np.ndarray):
        """Calculates cosine similarity between two DTRs.

        Args:
            x (np.ndarray): 1st DTR
            y (np.ndarray): 2nd DTR

        Returns:
            float: Cosine similarity between x and y.
        """
        similarity = np.dot(x,y)/(np.linalg.norm(x,2)*np.linalg.norm(y,2))
        return similarity

    def get_dtr_multifiles(self, 
                           img_path: List[str], 
                           angle: Union[None, int] = None, 
                           n_jobs: int = 8
                           ) -> np.ndarray:
        """Calculates DTRs for multiple images.

        Args:
            img_path (List[str]): List of image files.
            angle (Union[None, int], optional): Rotation angle (0-360). Defaults to None.
            n_jobs (int, optional): The number of parallel jobs used for preprocessing. Defaults to 8.

        Returns:
            np.ndarray: DTRs
        """
        #only preprocess runs in parallel
        imgs = Parallel(n_jobs=n_jobs)([delayed(self._process_image)(imgfile, angle) for imgfile in img_path])
        dtrs = np.empty((len(imgs), self.dim), astype=float)
        for i, img in enumerate(imgs):
            dtrs[i,:] = self.cbp([img])[0]

        return dtrs
