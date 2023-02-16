from typing import Dict, List, Union
import numpy as np
from sklearn.metrics import pairwise_distances as pair_dist
from PIL import Image
import matplotlib.pyplot as plt
import textwrap


def get_medoid(dtrs: np.ndarray,
               cases: List[str],
               ) -> Dict[str, int]:
    """Get medoid.

    Args:
        dtrs (np.ndarray): DTRs.
        cases (List[str]): Case IDs.

    Returns:
        Dict[str, int]: Dictionary containing case ID and the corresponding index of medoid image.
    """
    medoid_dict = {}
    u_cases = np.unique(cases)
    for case in u_cases:
        case_idx = np.where(np.asarray(cases) == case)[0]
        dmat = pair_dist(dtrs[case_idx, :],
                            metric='cosine')
        medoid_idx = case_idx[np.argmin(dmat.sum(axis=0))]
        medoid_dict[case] = medoid_idx

    return medoid_dict

def imgcats(infiles: List[str],
            labels: List[str],
            nrows: int = 3, 
            save: Union[None, str] = None,
            dpi: int = 320,
            fontsize: int = 12,
            w: int = 70,
            ) -> None:

    ncols = int(np.ceil(len(infiles)/nrows))
    for i, infile in enumerate(infiles):
        plt.subplot(ncols, nrows, i+1)
        im = Image.open(infile)
        im_list = np.asarray(im)
        plt.imshow(im_list)
        if len(labels) != 0:
            plt.title("\n".join(textwrap.wrap(labels[i], w)), fontsize=fontsize)
        plt.axis('off')
    if save != None:
        plt.savefig(save, dpi=dpi)
    plt.show()