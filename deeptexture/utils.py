from typing import Dict, List
import numpy as np
from sklearn.metrics import pairwise_distances as pair_dist


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
        dmat = pair_dist(X[case_idx, :],
                            metric='cosine')
        medoid_idx = case_idx[np.argmin(dmat.sum(axis=0))]
        medoid_dict[case] = medoid_idx

    return medoid_dict