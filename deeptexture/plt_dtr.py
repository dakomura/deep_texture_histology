from typing import Any, List, Union
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage,AnnotationBbox

from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np

from .utils import *

np.random.seed(42)
sns.set()
sns.set_style('ticks')

# plot image
def plt_dtr_image(X: np.ndarray,
                  files: List[str], 
                  method: Union[None, str] = None, 
                  dpi: int = 320, 
                  scale: float = 1.0, 
                  outfile: str = "", 
                  save: bool = False, 
                  axis: bool = False,
                  text:Union[None, List[str]] = None,
                  show_medoid: bool = False,
                  cases: Union[None, str] = None,
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
        show_medoid (bool, optional): only show medoid. Active if DTRs are given as X. Defaults to False.
        cases (Union[None, str], optional): Cases for each image. Valid only if show_medoid is True. Defaults to None.

    Returns:
        np.ndarray: Two-dimensional coordicates of DTRs for N images (Nx2 array) 
    """
    def _get_ab(imgfile, scale, x, y):
        img = Image.open(imgfile)
        width = img.width
        base_scale = 256/width

        im = OffsetImage(img, scale*base_scale)
        ab = AnnotationBbox(im, (x,y), xycoords='data', frameon=False)    
        return ab

    if len(files) != X.shape[0]:
        raise Exception("len(files) must be the number of dtrs.")

    if X.shape[1] == 2: # embedding
        X_emb = X
        print ("Embedding is given instead of DTRs")
    else: # dtr
        if method is None:
            raise Exception("Please specify `method` when `X` is not embedded values.")
        if not method in ["tsne", "pca", "umap", "lle", "isomap", "se"]:
            raise Exception("Please specify valid embedding method.")
        X_emb = _embed(X, method, **kwargs)

    if show_medoid:
        if cases is None or len(cases) != len(files):
            raise Exception("Invalid values for cases.")
        if X.shape[1] == 2:
            raise Exception("Show medoid is available only when DTRs are given.")
        #extract medoid for each case
        medoid_dict = get_medoid(X, cases)
            

    fig = plt.figure(figsize = (14, 10.5))
    ax = fig.add_subplot(1,1,1)

    if text is not None:
        if len(text) != len(files):
            raise Exception("len(text) must be the same as len(files).")
        for i,t in enumerate(text):
            ax.text(X_emb[i,0], X_emb[i,1], t, size=10,
                fontweight="bold")

    if show_medoid:
        for i, file in enumerate(files):
            if not i in medoid_dict.values():
                ab = _get_ab(file, 0.005*scale, X_emb[i,0],X_emb[i,1])
                ax.add_artist(ab)
        for i, file in enumerate(files):
            if i in medoid_dict.values():
                ab = _get_ab(file, 0.05*scale, X_emb[i,0],X_emb[i,1])
                ax.add_artist(ab)

    for i, file in enumerate(files):
        ab = _get_ab(file, 0.05*scale, X_emb[i,0],X_emb[i,1])
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
                 cases: Union[None, List[str]] = None,
                 method: Union[None, str] = None, 
                 s: int = 10,
                 outfile: str = "", 
                 palette: str = 'colorblind',
                 dpi: int = 320, 
                 save: bool = False, 
                 axis: bool = False,
                 text: Union[None, List[str]] = None,
                 use_plotly: bool = False,
                 show_medoid: bool = False,
                 **kwargs
                 ) -> pd.DataFrame:
    """Plot DTRs in two-dimensional space given or calculated by the specified dimensionality reduction method. 
    User specified attributes are plotted at the position in the space.

    Args:
        X (np.ndarray): M-dimensional DTRs for N images (NxM array) or two-dimensional coordicates of DTRs for N images (Nx2 array). 
        attr (List[str]): List of attribute string.
        cases (Union[None, str], optional): If not None, the case ID is shown or used for medoid calculation. Activated only when use_plotly is True or show_medoid is True. Defaults to None.
        method (Union[None, str], optional): If not None, the specified dimensionality reduction method ("tsne", "pca", "umap", "lle", "isomap", "se") is used. Defaults to None.
        s (int, optional): Point size in the plot. Defaults to 10.
        outfile (str, optional): Output image file. Defaults to "".
        palette (str, optional): Color palette. Defaults to 'colorblind'.
        dpi (int, optional): Dots per inch (DPI) of output image. Defaults to 320.
        save (bool, optional): Save the output image to outfile if True. Defaults to False.
        axis (bool, optional): Show axis if True. Defaults to False.
        use_plotly (bool, optional): Use plotly. Defaults to False.
        text (Union[None, List[str]], optional): Show text if not None. Defaults to Union[None, List[str]]. Defaults to None.
        show_medoid (bool, optional): only show medoid. Active if DTRs are given as X. Defaults to False.

    Returns:
        pd.DataFrame: Dataframe with 'attr', 'x1', and 'x2' columns.
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


    if show_medoid:
        s_list = np.ones((len(attr))) * 0.1 * s
        if cases is None or len(cases) != len(files):
            raise Exception("Invalid values for cases.")
        if X.shape[1] == 2:
            raise Exception("Show medoid is available only when DTRs are given.")
        #extract medoid for each case
        u_cases = list(set(cases))
        medoid_dict = get_medoid(X, cases)
    else:
        s_list = np.ones((len(attr))) * s
    

    is_medoid = [idx in medoid_dict.values() for idx in range(len(attr))]
    df = pd.DataFrame({'attr': attr,
                        'x1': X_emb[:,0],
                        'x2': X_emb[:,1],
                        'size': s_list,
                        'medoid':is_medoid})

    width1 = (np.max(X_emb[:,0]) - np.min(X_emb[:,0]))*0.15
    height1 = (np.max(X_emb[:,1]) - np.min(X_emb[:,1]))*0.15

    if use_plotly:
        import plotly.express as px
        fig = px.scatter(df, x='x1', y='x2',
                        color='attr',
                        size='size',
                        hover_name='attr',
                        hover_data = cases,
                        text=text,
                        range_x = [np.min(X_emb[:,0])-width1, np.max(X_emb[:,0])+width1],
                        lange_y = [np.min(X_emb[:,1])-height1, np.max(X_emb[:,1])+height1],
                        )
        if not axis:
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
        if save:
            print ("dpi is ignored in plotly mode.")
            fig.write_image(outfile)

    else:
        plt.clf()
        sns.scatterplot(x='x1', y='x2', data=df,
                    palette = palette,
                    s='size',
                    hue = 'attr')
        if text is not None:
            if len(text) != len(attr):
                raise Exception("len(text) must be the same as len(attributes).")
            for i,t in enumerate(text):
                plt.text(X_emb[i,0], X_emb[i,1], t, size=10,
                    fontweight="bold")

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
