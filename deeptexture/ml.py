from typing import Any, Union
import numpy as np
#from pycaret import classification as cl
import pandas as pd

class ML:
    def __init__(self,
                 dtrs: np.ndarray,
                 ) -> None:
        """initialize machine learning analysis using DTRs

        Args:
            dtrs (np.ndarray): M-dimensional DTRs for N images (NxM array)
        """
        
        self.dtrs = dtrs

    def get_caret_sl(self,
                     y: Union[list, np.ndarray],
                     cases: Union[list, np.ndarray],
                     y_name: str = 'target',
                     model: str = 'lr',
                     fold: int = 3,
                     pca: bool = False,
                     ) -> Any:
        """Supervised learning model using pycaret

        Args:
            y (Union[list, np.ndarray]): Target variable.
            cases (Union[list, np.ndarray]): Case IDs (used as group).
            y_name (str, optional): Name of target variable. Defaults to 'target'.
            model (str, optional): Supervised learning model. Defaults to 'lr'.
            fold (int, optional): Number of folds in K-fold cross validation. Defaults to 3.
            pca (bool, optional): Apply PCA for preprocessing. Defaults to False.

        Returns:
            Any: Tuned model.
        """

        ndim = self.dtrs.shape[1]
        digit = len(str(ndim))
        
        df = pd.DataFrame(data=self.dtrs,
                          coluns=['dim{}'.format(str(i).zfill(digit)) for i in range(ndim)],
                          )
        df[y_name] = y

        return df #tuned_model