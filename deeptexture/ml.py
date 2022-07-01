from typing import Any, Union
import numpy as np
from pycaret import classification as cl
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

        ndim = self.dtrs.shape[1]
        digit = len(str(ndim))
        
        df = pd.DataFrame(data=self.dtrs,
                          coluns=['dim{}'.format(str(i).zfill(digit)) for i in range(ndim)],
                          )
        df[y_name] = y

        s = cl.setup(df, 
                     target = y_name, 
                     fold_strategy = 'groupkfold',
                     fold_groups = cases,
                     fold = fold,
                     pca = pca,
                     session_id = 123,
                     )
        model = cl.create_model(model)
        print(model)

        print("model tuning")
        tuned_model = cl.tune_model(model)

        pred = cl.predict_model(tuned_model,
                                verbose = True,
                                )

        return tuned_model