from typing import Any, List, Union
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
                  text:Union[None, List[str]] = None,
                  **kwargs,
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
        text (Union[None, List[str]], optional): Show text if not None. Defaults to Union[None, List[str]]. Defaults to None.

    Returns:
        np.ndarray: Two-dimensional coordicates of DTRs for N images (Nx2 array) 
    """
    if len(files) != X.shape[0]:
        raise Exception("len(files) must be the number of dtrs.")

    if X.shape[1] == 2: # embedding
        X_emb = X
        print ("Embedding is given instead of DTRs")
    else: # dtr
        if method is None:
            raise Exception("Please specify `method` when `X` is not embedded values")
        if not method in ["tsne", "pca", "umap", "lle", "isomap", "se"]:
            raise Exception("Please specify valid embedding method")
        X_emb = _embed(X, method, **kwargs)

    fig = plt.figure(figsize = (14, 10.5))
    ax = fig.add_subplot(1,1,1)

    if text is not None:
        if len(text) != len(files):
            raise Exception("len(text) must be the same as len(files).")
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
                 axis: bool = False,
                 text: Union[None, List[str]] = None,
                 **kwargs
                 ) -> Any:
    """Plot DTRs in two-dimensional space given or calculated by the specified dimensionality reduction method. 
    User specified attributes are plotted at the position in the space.

    Args:
        X (np.ndarray): M-dimensional DTRs for N images (NxM array) or two-dimensional coordicates of DTRs for N images (Nx2 array). 
        attr (List[str]): List of attribute string.
        method (Union[None, str], optional): If not None, the specified dimensionality reduction method ("tsne", "pca", "umap", "lle", "isomap", "se") is used. Defaults to None.
        s (int, optional): Point size in the plot. Defaults to 10.
        outfile (str, optional): Output image file. Defaults to "".
        palette (str, optional): Color palette. Defaults to 'colorblind'.
        dpi (int, optional): Dots per inch (DPI) of output image. Defaults to 320.
        save (bool, optional): Save the output image to outfile if True. Defaults to False.
        axis (bool, optional): Show axis if True. Defaults to False.
        text (Union[None, List[str]], optional): Show text if not None. Defaults to Union[None, List[str]]. Defaults to None.

    Returns:
        Any: Pandas dataframe with 'attr', 'x1', and 'x2' columns.
    """
    if len(attr) != X.shape[0]:
        raise Exception("len(attributes) must be the number of dtrs.")

    if X.shape[1] == 2: # embedding
        X_emb = X
        print ("Embedding is given instead of DTRs")
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
    if text is not None:
        if len(text) != len(attr):
            raise Exception("len(text) must be the same as len(attributes).")
        for i,t in enumerate(text):
            plt.text(X_emb[i,0], X_emb[i,1], t, size=10,
                fontweight="bold")

    width1 = (np.max(X_emb[:,0]) - np.min(X_emb[:,0]))*0.15
    height1 = (np.max(X_emb[:,1]) - np.min(X_emb[:,1]))*0.15
    plt.xlim(np.min(X_emb[:,0])-width1, np.max(X_emb[:,0])+width1)
    plt.ylim(np.min(X_emb[:,1])-height1, np.max(X_emb[:,1])+height1)

    if not axis:
        plt.axis('off')

    if save:
        plt.savefig(outfile,dpi = dpi)

    return df

def _embed(X: np.ndarray, 
          method: str, 
          **kwargs
          ) -> np.ndarray:

    if method == 'tsne':
        from sklearn.manifold import TSNE
        p = kwargs.get('p', 100)
        print (f"perplexity {p} for TSNE plot.")
        return TSNE(n_components = 2,
                    perplexity = p,
                    n_iter = 1000,
                    random_state = 42).fit_transform(X)
    if method == 'pca':
        from sklearn.decomposition import IncrementalPCA
        x1 = kwargs.get("x1", 1)
        x2 = kwargs.get("x2", 2)
        if x1 <= 0 or x2 <= 0:
            raise Exception("invalid dimensions for PCA")
        print (f"PC{x1} and PC{x2} are used for PCA plot.")
        pca = IncrementalPCA(n_components = max(x1, x2),
                    batch_size = 100,
                    whiten = True)
        pca.fit(X)
        X_emb = pca.transform(X)
        X_emb_new = np.empty((X_emb.shape[0], 2))
        X_emb_new[:,0] = X_emb[:,x1-1]
        X_emb_new[:,1] = X_emb[:,x2-1]
        return X_emb_new
    if method == 'umap':
        import umap
        min_dist = kwargs.get('min_dist', 0.1)
        n_neighbors = kwargs.get('n_neighbors', 5)
        metric = kwargs.get('metric', "cosine")
        print (f"min_dist: {min_dist}, n_neighbors: {n_neighbors}, and metric: {metric} are used for UMAP plot")
        return umap.UMAP(min_dist = min_dist,
                         n_neighbors = n_neighbors,
                         metric = metric).fit_transform(X)
    if method == 'lle':
        from sklearn.manifold import LocallyLinearEmbedding as LLE
        # LLE method = 'standard', 'ltsa', 'hessian', 'modified'
        return LLE(method='modified',**kwargs).fit_transform(X)
    if method == 'isomap':
        from sklearn.manifold import Isomap
        return Isomap(**kwargs).fit_transform(X)
    if method == 'se':
        from sklearn.manifold import SpectralEmbedding as SE
        return SE(**kwargs).fit_transform(X)
