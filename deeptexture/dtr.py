from typing import Any, List, Union
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

        print(f"arch:{arch}")
        print(f"layer:{layer}")
        print(f"dim:{dim}")

        self._create_model()

        # preprocess function
        self.prep = self.prep_dict[arch].preprocess_input


    def _create_model(self) -> None:

        conv_base = self.archs_dict[self.arch](
            weights = "imagenet",
            include_top = None,
            input_shape = (None, None, 3))

        x1 = conv_base.get_layer(self.layer).output
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
            x = np.array(img)
        elif not type(img) == np.ndarray:
            x = np.array(img)
        else:
            x = img

        if angle is not None:
            x = preprocessing.image.apply_affine_transform(x, theta = angle)

        x = np.expand_dims(x, axis=0)
        x = self.prep(x)
        return np.array(self.cbp([x])[0])

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
                           ) -> np.ndarray:
        """Calculates DTRs for multiple images.

        Args:
            img_path (List[str]): List of image files.
            angle (Union[None, int], optional): Rotation angle (0-360). Defaults to None.

        Returns:
            np.ndarray: DTRs
        """
        dtrs = np.vstack([self.get_dtr(imgfile) for imgfile in img_path])
    
        return dtrs
