from typing import Any, List, Tuple, Union
from PIL import Image
import numpy as np
import cv2
import pandas as pd

from .utils import *

import torch

class DTR:
    def __init__(self,
                 vision_model: str = "ViT-L/14@336px",
                 ) -> None:
        """Initialize CLIP model

        Args:
            vision_model (str, optional): Image vision model (RN50,RN101,RN50x4,RN50x64,ViT-B/32,ViT-B/16,ViT-L/14,ViT-L/14@336px). Defaults to "ViT-L/14@336px".
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_model = vision_model
        self._create_model()

    def _create_model(self) -> None:
        import clip
        self.model, self.preprocess = clip.load(self.vision_model, device=self.device)

    def get_dtr(self, 
                img: Any, 
                angle: Union[None, int, List[int]] = None,
                ) -> np.ndarray:
        """Calculates DTR for an image object or file.

        Args:
            img (Any): Image file or image object (numpy array or PIL Image object)
            angle (Union[None, int, List[int]], optional): Rotation angle(s) (0-360). If list is given, mean DTRs of the rotated image return. Defaults to None.

        Returns:
            np.ndarray: DTR for the image
        """
        if type(img) == str:
            img = Image.open(img)
        elif type(img) == np.ndarray:
            img = Image.fromarray(img)


        if angle is not None:
            if type(angle) == int:
                img = img.rotate(angle)
            elif type(angle) == list:
                dtrs = np.vstack([self.get_dtr(img, theta) for theta in angle])
                dtr_mean = np.mean(dtrs, axis=0)
                return dtr_mean / np.linalg.norm(dtr_mean, ord=2) #L2-normalize
            else:
                raise Exception(f"invalid data type in angle {angle}")
                
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = np.array(image_features.cpu().squeeze())

        return image_features

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
        
        
