from typing import Any, List, Union
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import SpectralEmbedding as SE
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage,AnnotationBbox

from PIL import Image

import seaborn as sns
import pandas as pd
import numpy as np

np.random.seed(42)
sns.set()
sns.set_style('ticks')

#画像をプロットする
def plt_dtr_image(X: np.ndarray,
                  files: List[str], 
                  method: Union[None, str] = None, 
                  dpi: int = 320, 
                  scale: float = 1.0, 
                  outfile: str = "", 
                  save: bool = False, 
                  axis: bool = False,
                  text = Union[None, List[str]],
                  **kwargs
                  ) -> np.ndarray:
    """Plot DTRs in two-dimensional space given or calculated by the specified dimensionality reduction method. 
    Images are plotted at the position in the space.

    Args:
        X (np.ndarray): M-dimensional DTRs for N images (NxM array) or two-dimensional coordicates of DTRs for N images (Nx2 array). 
        files (List[str]): list of image file.
        method (Union[None, str], optional): If not None, the specified dimensionality reduction method ("tsne", "pca", "umap", "lle", "isomap", "se") is used. Defaults to None.
        dpi (int, optional): Dots per inch (DPI) of output image. Defaults to 320.
        scale (float, optional): Image scale in the plot. Defaults to 1.0.
        outfile (str, optional): Output image file. Defaults to "".
        save (bool, optional): Save the output image to outfile if True. Defaults to False.
        axis (bool, optional): Show axis if True. Defaults to False.
        text (_type_, optional): Show text if not None. Defaults to Union[None, List[str]].

    Returns:
        np.ndarray: _description_
    """

    if X.shape[1] == 2: # embedding
        X_emb = X
    else: # dtr
        if method is None:
            raise Exception("Please specify `method` when `X` is not embedded values")
        if not method in ["tsne", "pca", "umap", "lle", "isomap", "se"]:
            raise Exception("Please specify valid embedding method")
        X_emb = _embed(X, method, **kwargs)

    fig = plt.figure(figsize = (14, 10.5))
    ax = fig.add_subplot(1,1,1)

    if text is not None:
        for i,t in enumerate(text):
            ax.text(X_emb[i,0], X_emb[i,1], t, size=10,
                fontweight="bold")

    for i, file in enumerate(files):
        img = Image.open(file)
        width = img.width
        base_scale = 256/width

        im = OffsetImage(img, 0.05*scale*base_scale)
        ab = AnnotationBbox(im, (X_emb[i,0],X_emb[i,1]), xycoords='data', frameon=False)
        ax.add_artist(ab)
            
    width1 = (np.max(X_emb[:,0]) - np.min(X_emb[:,0]))*0.1
    height1 = (np.max(X_emb[:,1]) - np.min(X_emb[:,1]))*0.1

    plt.xlim(np.min(X_emb[:,0])-width1, np.max(X_emb[:,0])+width1)
    plt.ylim(np.min(X_emb[:,1])-height1, np.max(X_emb[:,1])+height1)
    plt.tight_layout()

    if not axis:
        ax.axis('off')

    if save:
        if len(outfile) == 0:
            raise Exception("Please specify output file")
        plt.savefig(outfile,dpi = dpi)

    return X_emb


def plt_dtr_attr(X: np.ndarray,
                 attr: List[str],
                 method: Union[None, str] = None, 
                 s: int = 10,
                 outfile: str = "", 
                 palette: str = 'colorblind',
                 dpi: int = 320, 
                 save: bool = False, 
                 **kwargs
                 ) -> Any:
    """_summary_

    Args:
        X (np.ndarray): M-dimensional DTRs for N images (NxM array) or two-dimensional coordicates of DTRs for N images (Nx2 array). 
        attr (List[str]): List of attribute string.
        method (Union[None, str], optional): If not None, the specified dimensionality reduction method ("tsne", "pca", "umap", "lle", "isomap", "se") is used. Defaults to None.
        s (int, optional): Point size in the plot. Defaults to 10.
        outfile (str, optional): Output image file. Defaults to "".
        palette (str, optional): Color palette. Defaults to 'colorblind'.
        dpi (int, optional): Dots per inch (DPI) of output image. Defaults to 320.
        save (bool, optional): Save the output image to outfile if True. Defaults to False.

    Returns:
        Any: Pandas dataframe with 'attr', 'x1', and 'x2' columns.
    """

    if X.shape[1] == 2: # embedding
        X_emb = X
    else: # dtr
        if method is None:
            raise Exception("Please specify `method` when `X` is not embedded values")
        if not method in ["tsne", "pca", "umap", "lle", "isomap", "se"]:
            raise Exception("Please specify valid embedding method")
        X_emb = _embed(X, method, **kwargs)

    df = pd.DataFrame({'attr': attr,
                        'x1': X_emb[:,0],
                        'x2': X_emb[:,1]})
    plt.clf()
    sns.scatterplot(x='x1', y='x2', data=df,
                palette = palette,
                s=s,
                hue = 'attr')

    width1 = (np.max(X_emb[:,0]) - np.min(X_emb[:,0]))*0.15
    height1 = (np.max(X_emb[:,1]) - np.min(X_emb[:,1]))*0.15
    plt.xlim(np.min(X_emb[:,0])-width1, np.max(X_emb[:,0])+width1)
    plt.ylim(np.min(X_emb[:,1])-height1, np.max(X_emb[:,1])+height1)
    if save:
        plt.savefig(outfile,dpi = dpi)

    return df

def _embed(X: np.ndarray, 
          method: str, 
          **kwargs
          ) -> np.ndarray:

    if method == 'tsne':
        return TSNE(n_components = 2,
                    perplexity = kwargs['p'],
                    n_iter = 1000,
                    random_state = 42).fit_transform(X)
    if method == 'pca':
        pca = IncrementalPCA(n_components = max(kwargs['x1'], kwargs['x2'])+1,
                    batch_size = 100,
                    whiten = True)
        pca.fit(X)
        X_emb = pca.transform(X)
        X_emb_new = np.empty((X_emb.shape[0], 2))
        X_emb_new[:,0] = X_emb[:,kwargs['x1']]
        X_emb_new[:,1] = X_emb[:,kwargs['x2']]
        return X_emb_new
    if method == 'umap':
        import umap
        return umap.UMAP(**kwargs).fit_transform(X)
    if method == 'lle':
        # LLE method = 'standard', 'ltsa', 'hessian', 'modified'
        return LLE(method='modified',**kwargs).fit_transform(X)
    if method == 'isomap':
        return Isomap(**kwargs).fit_transform(X)
    if method == 'se':
        return SE(**kwargs).fit_transform(X)
