from typing import Any, List, Tuple, Union
from PIL import Image
from pyrsistent import mutant
import numpy as np
import cv2
import pandas as pd
from .utils import *

import timm 
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import torch.nn.functional as F
from torchvision import transforms


class DTR:
    def __init__(self,
                 arch: str = 'resnetrs350',
                 layer: int = 2, 
                 ) -> None:
        """Initializes DTR model and its preprocessing function.

        Args:
            arch (str, optional): CNN model. Defaults to 'vgg'.
            layer (str, optional): A layer in the CNN model. Defaults to 'block4_conv3'.
            dim (int, optional): The output dimension. Defaults to 1024.
        """
        self.arch = arch
        self.layer = layer

        print(f"arch:{arch}")
        print(f"layer:{layer}")

        self._create_model()

        # preprocess function
        self.config = resolve_data_config({}, model=self.model)
        self.prep = create_transform(**self.config)


    def _create_model(self) -> None:
        self.model = timm.create_model(
            self.arch,
            features_only=True,
            pretrained=True,
        ).to('cuda')
        self.model.eval()
        self.layers = len(self.model.feature_info.channels())

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
            # Filename
            ximg = Image.open(img).convert("RGB")
        elif not type(img) == np.ndarray:
            # Image
            ximg = img
        else: #numpy
            ximg = Image.fromarray(img)

        if size is not None:
            x = ximg.resize((size, size))
        elif scale is not None:
            h, w = ximg.size
            x = ximg.resize((int(h*scale), int(w*scale)))

        if multi_scale:
            #1/4 scale
            h, w = x.size
            x2 = x.resize((int(h*0.25), int(w*0.25)))
            x2 = self.prep(x2).unsqueeze(0).to('cuda')
            x2 = x2.permute(0,3,1,2)

        # To Tensor (batch, channel, H, W)
        x = self.prep(x).unsqueeze(0).to('cuda')
        x = x.permute(0,3,1,2)

        if angle is not None:
            if type(angle) == int:
                x = transforms.functional.rotate(x, angle = angle)
                if multi_scale: 
                    x2 = transforms.functional.rotate(x2, angle = angle)
            elif type(angle) == list:
                dtrs = np.vstack([self.get_dtr(ximg, theta, size, scale, multi_scale) for theta in angle])
                dtr_mean = np.mean(dtrs, axis=0)
                return dtr_mean / np.linalg.norm(dtr_mean, ord=2) #L2-normalize
            else:
                raise Exception(f"invalid data type in angle {angle}")
                
        dtr = self.forward(x)

        if multi_scale:
            dtr2 = self.forward(x2)
            dtr = np.concatenate([dtr, dtr2])

        return dtr

    def forward(self, x):
        output = self.model(x)[self.layer]
        t_torch = F.avg_pool2d(output, kernel_size=output.shape[-1]) 
        dtr = t_torch.squeeze().to('cpu').detach().clone().numpy()
        dtr = dtr / np.linalg.norm(dtr, ord=2) #L2-normalize

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
        
        
