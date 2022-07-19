from typing import Any, List, Tuple, Union
from PIL import Image
from pyrsistent import mutant
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, preprocessing
from tensorflow.keras.applications import resnet50, vgg16, mobilenet_v2, inception_v3, nasnet, densenet, inception_resnet_v2

from .utils import *

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
                angle: Union[None, int, List[int]] = None,
                size: Union[None, int] = None,
                scale: Union[None, float] = None,
                multi_scale: bool = False,
                ) -> np.ndarray:
        """Calculates DTR for an image object or file.

        Args:
            img (Any): Image file or image object (numpy array or PIL Image object)
            angle (Union[None, int, List[int]], optional): Rotation angle(s) (0-360). If list is given, mean DTRs of the rotated image return. Defaults to None.
            size (Union[None, int], optional): Image is resized to the given size. Default to None.
            scale (Union[None, int], optional): Image is rescaled. Active only size is not specified. Default to None.
            multi_scale (bool, optional): DTR for 1/4 sized image is concatenated. The dimension of the DTR will be  2*dim. Default to False.

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

        if size is not None:
            x = cv2.resize(x, dsize=[size, size])
        elif scale is not None:
            h, w, _ = x.shape
            x = cv2.resize(x, dsize=[int(h*scale), int(w*scale)])

        if multi_scale:
            #1/4 scale
            x2 = cv2.resize(x, dize=None, fx=0.25, fy=0.25)

        if angle is not None:
            if type(angle) == int:
                x = preprocessing.image.apply_affine_transform(x, theta = angle)
                if multi_scale: 
                    x2 = preprocessing.image.apply_affine_transform(x2, theta = angle)
            elif type(angle) == list:
                dtrs = np.vstack([self.get_dtr(img, theta, size, scale, multi_scale) for theta in angle])
                dtr_mean = np.mean(dtrs, axis=0)
                return dtr_mean / np.linalg.norm(dtr_mean, ord=2) #L2-normalize
            else:
                raise Exception(f"invalid data type in angle {angle}")
                

        x = np.expand_dims(x, axis=0)
        x = self.prep(x)
        dtr = np.array(self.cbp([x])[0])

        if multi_scale:
            x2 = np.expand_dims(x2, axis=0)
            x2 = self.prep(x2)
            dtr2 = np.array(self.cbp([x2])[0])
            dtr = np.concatenate([dtr, dtr2])

        return dtr

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
                           imgfiles: List[str], 
                           angle: Union[None, int, List[int]] = None, 
                           size: Union[None, int] = None,
                           scale: Union[None, float] = None,
                           ) -> np.ndarray:
        """Calculates DTRs for multiple images.

        Args:
            imgfiles (List[str]): List of image files.
            angle (Union[None, int, List[int]], optional): Rotation angle(s) (0-360). If list is given, mean DTRs of the rotated image return. Defaults to None.
            size (Union[None, int], optional): Image is resized to the given size. Default to None.
            scale (Union[None, int], optional): Image is rescaled. Active only size is not specified. Default to None.

        Returns:
            np.ndarray: DTRs
        """
        dtrs = np.vstack([self.get_dtr(imgfile, angle=angle, size=size, scale=scale) for imgfile in imgfiles])
    
        return dtrs
    
    def get_mean_dtrs(self,
                      dtrs: np.ndarray,
                      cases: List[str],
                      df: Union[None, pd.DataFrame, List[str]] = None,
                      ) -> Tuple[np.ndarray, List[str], Union[None, pd.DataFrame, List[str]]]:
        """Calculate mean dtrs.

        Args:
            dtrs (np.ndarray): M-dimensional DTRs for N images (NxM array).
            cases (List[str]): List of case IDs for N images.
            df: Union[None, pd.DataFrame, List[str]]: List or dataframe containing attributes of N images. The order should be the same as dtrs and cases. Default to None.

        Returns:
            Tuple[np.ndarray, List[str], List[str]]: mean DTRs, 
            List of image file path of the representative images (medoid for each case), and case IDs.
        """

        
        u_cases = np.sort(np.unique(cases))
        dtrs_mean = np.vstack([np.mean(dtrs[np.array(cases)==case, :], axis=0) for case in u_cases])
        dtrs_mean = dtrs_mean / np.linalg.norm(dtrs_mean, ord=2) #L2-normalize

        medoid_dict = get_medoid(dtrs, cases)

        if df is not None:
            if type(df) == list:
                df_mean =[df[medoid_dict[case]] for case in u_cases]
            else:
                #df = df.reset_index()
                idx = np.array([medoid_dict[case] for case in u_cases])
                df_mean = df.iloc[idx, :]
        else:
            df_mean = None
                
        return dtrs_mean, list(u_cases), df_mean
        
        
