from audioop import add
from typing import Any, List, Union
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import *

class ML:
    def __init__(self,
                 dtrs: np.ndarray,
                 imgfiles: List[str],
                 ) -> None:
        """initialize machine learning analysis using DTRs

        Args:
            dtrs (np.ndarray): M-dimensional DTRs for N images (NxM array)
            imgfiles (List[str]): List of N image files.
        """
        
        self.dtrs = dtrs
        self.imgfiles = imgfiles

    def fit_eval(self,
               y: Union[list, np.ndarray],
               cases: Union[list, np.ndarray],
               additional_features: np.ndarray = None,
               min_samples: int = 5,
               show: bool = True,
               ) -> Any:
        """Logistic regression analysis.

        Args:
            y (Union[list, np.ndarray]): Target variable.
            cases (Union[list, np.ndarray]): Case IDs (used as group).
            additional_features (np.ndarray, optional): Additional features used for the classification. It MUST be the numerical arrays. If it is a categorical variable, please use categorical encoders. Defaults to None.
            min_samples (int, optional): Minimum number of cases analyzed in a target. Targets below the value will be removed. Defaults to 5.
            show (bool, optional): Show confusion matrix or ROC curve. Defaults to True.
        Returns:
            Any: AUROC (for binary classification) or confusion matrix (for multiclass classification).
        """

        
        #count cases for each class
        labels = np.unique(y)
        used_index = []
        for l in labels:
            target_index = np.where(np.array(y) == l)[0]
            count = len(np.unique(np.array(cases)[np.array(y) == l]))
            if count < min_samples:
                print(f'class {l} is not analyzed (only {count} cases)')
            else:
                used_index.extend(list(target_index))
        
        dtrs2 = self.dtrs[used_index,:]
        if additional_features is not None:
            if len(additional_features.shape) == 1:
                additional_features = np.expand_dims(additional_features, axis=1)
            dtrs2 = np.concatenate([dtrs2, additional_features],axis=1)
        y = np.array(y)[used_index]
        cases = np.array(cases)[used_index]
        labels = list(np.unique(y))
        print(f'labels: {labels}')

        if len(labels) > 2:
            mode = 'multi'
            print ("Muticlass => SVM")
            model = SVC(kernel = 'linear', C = 1)
        elif len(labels) == 2:
            mode = 'binary'
            print ("Binary => Logistic Regression")
            model =  LogisticRegression(solver='liblinear') 
        else:
            raise Exception(f'invalid number of classes ({labels})')


        #split cases
        u_cases = np.unique(cases)
        y_ucases = [y[cases==c][0] for c in u_cases] 
        train_cases, test_cases = train_test_split(u_cases, 
                                                   test_size=0.25,
                                                   stratify=y_ucases,
                                                   random_state=0) 
        X_train = np.vstack([x for i, x in enumerate(dtrs2) if cases[i] in train_cases])
        X_test = np.vstack([x for i, x in enumerate(dtrs2) if cases[i] in test_cases])
        y_train = [x for i, x in enumerate(y) if cases[i] in train_cases]
        y_test = [x for i, x in enumerate(y) if cases[i] in test_cases]


        model.fit(X_train, y_train) 
        self.model = model
        
        if mode == 'multi':
            y_pred = model.predict(X_test)
            conf_mat = confusion_matrix(y_test, y_pred, labels=labels)
            conf_mat = pd.DataFrame(data=conf_mat,
                                    index=labels,
                                    columns=labels)
            if show:
                g=sns.heatmap(conf_mat, square=True, cbar=True, annot=True, cmap='Blues', fmt="d")
                g.set_xticklabels(g.get_xticklabels(), rotation=90, horizontalalignment='right')
                g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')
                plt.xlabel("Prediction", fontsize=13, rotation=0)
                plt.ylabel("Ground Truth", fontsize=13)

            return conf_mat
        else:
            probs = model.predict_proba(X_test)
            preds = probs[:,1]
            fpr, tpr, _ = metrics.roc_curve(y_test, preds)
            roc_auc = metrics.auc(fpr, tpr)

            if show:
                plt.title('Receiver Operating Characteristic')
                plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
                plt.legend(loc = 'lower right')
                plt.plot([0, 1], [0, 1],'r--')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.show()

            return roc_auc


    def clustering(self,
                   method: str = 'bayes_gmm',
                   n_components: int = 10,
                   show: bool = False,
                   ) -> List[int]:
        """Clustering of dtrs.

        Args:
            method (str, optional): Clustering algorithm. Defaults to 'bayes_gmm'.
            n_components (int, optional): Number of (maximum) clusters. Defaults to 10.
            show (bool, optional): Show representative images. Defaults to False.

        Returns:
            List[int]: Cluster labels.
        """
                   

        if method == 'bayes_gmm':
            from sklearn.mixture import BayesianGaussianMixture
            model = BayesianGaussianMixture(n_components=n_components,
                                            random_state=42)
        else:
            raise Exception(f'invalid clustering algorithm: {method}')

        cluster_label = model.fit_predict(self.dtrs)

        if show:
            medoid_dict = get_medoid(self.dtrs, cluster_label)
            imgfiles_medoid = [self.imgfiles[medoid_dict[c]] for c in sorted(np.unique(cluster_label))]
            imgcats(imgfiles_medoid, 
                    labels = sorted(np.unique(cluster_label)),
                    nrows = 3)

        return cluster_label