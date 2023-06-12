from typing import Any, List, Tuple, Union
from PIL import Image
from pyrsistent import mutant
import numpy as np
import cv2
import pandas as pd
import os

import torch
from torchvision.models import vgg16, VGG16_Weights, resnet18
from collections import namedtuple
from torchvision import transforms
from scipy.ndimage import rotate
import torch.nn.functional as F
import pytorch_lightning as pl
from lightly.models.modules.heads import TiCoProjectionHead


from .utils import *

class TiCo(pl.LightningModule):
    def __init__(self, backbone, hidden_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = TiCoProjectionHead(hidden_dim, hidden_dim, out_dim)
        #self.criterion = TiCoLoss()


class DTR():
    def __init__(self, 
                 arch: str = 'vgg', #only vgg is supported
                 model_dir: str = None,
                 layer: str = 'block3_conv3',
                 dim: int = 1024,
                 rand_1: Union[int, None] = None, 
                 rand_2: Union[int, None] = None, 
                 device: Union[int, str] = 'cuda:0',
                 ):

        self.arch = arch
        self.device = device

        if arch == 'vgg':       
            self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
            self.model.eval()

            if layer == 'block3_conv3':
                self.i = 16
            elif layer == 'block4_conv3':
                self.i = 23
            else:
                raise ("Only block3_conv3 or block4_conv3 layers are supported in this version.")


            # 新しいフォワード関数をモデルに適用
            self.model.forward = self.new_forward.__get__(self.model, self.model.__class__)

            self.features_list = list(self.model.features)[:self.i]
            self.input_dim = self.features_list[-2].out_channels

            self.output_dim = dim

            if rand_1 is None:
                np.random.seed(128)
                rand_1 = np.random.randint(2,size=(self.input_dim, self.output_dim))*2-1
                self.rand_1 = torch.Tensor(rand_1).to(device)
                
            if rand_2 is None:
                np.random.seed(1997)
                rand_2 = np.random.randint(2,size=(self.input_dim, self.output_dim))*2-1
                self.rand_2 = torch.Tensor(rand_2).to(device)

        elif arch == 'tico':
            resnet = resnet18()
            resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.model = TiCo.load_from_checkpoint(os.path.join(model_dir, "TiCo_SSL_model.pt"), 
                                                   backbone=resnet, hidden_dim=512, out_dim=128)
            self.model = self.model.backbone
            self.model.eval()            

        else:
            raise ("Only VGG16 or SSL TiCo model are supported in this version.")

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            )

        self.prep = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

        self.model = self.model.to(self.device)

    # 特定の中間層の出力を取得するための新しいフォワード関数を定義
    def new_forward(self, x):
        for layer in self.model.features[:self.i]:
            x = layer(x)
        return x
        
    def forward(self, bottom):
        """
        bottom: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width].
        """
        assert bottom.size(1) == self.input_dim

        batch_size, _, height, width = bottom.size() #128,512,14,14
        
        bottom_flat = bottom.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim) #128*14*14,512
        
        bottom1_mat = torch.matmul(bottom_flat,self.rand_1)
        bottom2_mat = torch.matmul(bottom_flat,self.rand_2)
        
        cbp_flat = bottom1_mat * bottom2_mat
        
        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)
        cbp = cbp.mean(dim=1).mean(dim=1)

        cbp = torch.sqrt(torch.abs(cbp)) * torch.sign(cbp)
        cbp = F.normalize(cbp, p=2, dim=1)
        
        return cbp

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
                x = rotate(x, angle = angle)
                if multi_scale: 
                    x2 = rotate(x2, angle = angle)
            elif type(angle) == list:
                dtrs = np.vstack([self.get_dtr(img, theta, size, scale, multi_scale) for theta in angle])
                dtr_mean = np.mean(dtrs, axis=0)
                return dtr_mean / np.linalg.norm(dtr_mean, ord=2) #L2-normalize
            else:
                raise Exception(f"invalid data type in angle {angle}")
                

        x = self.prep(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.arch == 'vgg':
                x = self.forward(self.model(x))
            elif self.arch == 'tico':
                x = self.model(x)
            dtr = x.cpu().detach().numpy()

        if multi_scale:
            x2 = self.prep(x2).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if self.arch == 'vgg':
                    x2 = self.forward(self.model(x2))
                elif self.arch == 'tico':
                    x2 = self.model(x)
                dtr2 = x2.cpu().detach().numpy()
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
        
