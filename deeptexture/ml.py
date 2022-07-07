from typing import Any, Union
import numpy as np
#from pycaret import classification as cl
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ML:
    def __init__(self,
                 dtrs: np.ndarray,
                 ) -> None:
        """initialize machine learning analysis using DTRs

        Args:
            dtrs (np.ndarray): M-dimensional DTRs for N images (NxM array)
        """
        
        self.dtrs = dtrs

    def get_lr(self,
               y: Union[list, np.ndarray],
               cases: Union[list, np.ndarray],
               ) -> Any:
        """Logistic regression analysis.

        Args:
            y (Union[list, np.ndarray]): Target variable.
            cases (Union[list, np.ndarray]): Case IDs (used as group).
        Returns:
            Any: Logistic Regression model.
        """

        if len(np.unique(y)) > 2:
            mode = 'multi'
            print ("Muticlass => SVM")
            model = SVC(kernel = 'linear', C = 1)
        else:
            mode = 'binary'
            print ("Binary => Logistic Regression")
            model =  LogisticRegression(solver='liblinear') 

        #split cases
        u_cases = np.unique(cases)
        train_cases, test_cases = train_test_split(u_cases, 
                                                   test_size=0.25,
                                                   random_state=0) 
        X_train = np.vstack([x for i, x in enumerate(self.dtrs) if cases[i] in train_cases])
        X_test = np.vstack([x for i, x in enumerate(self.dtrs) if cases[i] in test_cases])
        y_train = [x for i, x in enumerate(y) if cases[i] in train_cases]
        y_test = [x for i, x in enumerate(y) if cases[i] in test_cases]


        model.fit(X_train, y_train) 
        
        if mode == 'multi':
            y_pred = model.predict(X_test)
            conf_mat = confusion_matrix(y_test,y_pred)
            sns.heatmap(conf_mat, square=True, cbar=True, annot=True, cmap='Blues')
        else:
            probs = model.predict_proba(X_test)
            preds = probs[:,1]
            fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
            roc_auc = metrics.auc(fpr, tpr)

            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()


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
                          columns=['dim{}'.format(str(i).zfill(digit)) for i in range(ndim)],
                          )
        df[y_name] = y

        outstring = "s = cl.setup(df,\n" + \
            f"            target = '{y_name}',\n" + \
            "            fold_strategy = 'groupkfold',\n" + \
            "            fold_groups = cases,\n" + \
            f"            fold = {fold},\n" + \
            f"            pca = {pca},\n" + \
            "            session_id = 123,\n" + \
            "            )\n" + \
            f"ml_model = cl.create_model({model})\n" + \
            "tuned_model = cl.tune_model(ml_model)\n" + \
            "pred = cl.predict_model(tuned_model,\n" + \
            "                        verbose = True)" 
        print(outstring)
        return df, cases