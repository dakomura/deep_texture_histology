from functools import reduce
import os
from typing import Any, List
import nmslib
import joblib
import numpy as np
import collections
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from .dtr import *
from .utils import *
plt.rcParams['font.size'] = 10 #font size
plt.rcParams['figure.figsize'] = 20,70 #figure size

class CBIR:
    def __init__(self,
                    dtr_obj: Any,
                    project: str = "DB",
                    working_dir: str = ".",
                    ) -> None:
        """Initialize an object for Content-based Image Retrieval using DTRs.

        Args:
            dtr_obj (Any): DTR object.
            project (str, optional): Database name. This value is used as a directory name in which all the data is stored. Defaults to "DB".
            working_dir (str, optional): Working directory. All the data is stored under the directory. Defaults to ".".
        """
        self.dtr_obj = dtr_obj
        self.project = project
        self.working_dir = working_dir
        self.indexfile = '{}/{}/nmslib_index.idx'.format(self.working_dir,
                                                            self.project)
        os.makedirs(os.path.dirname(self.indexfile), exist_ok=True)

    def create_db(self,
                    df_attr: Any,
                    dtrs: Union[None, np.ndarray] = None,
                    img_attr: str = "imgfile",
                    case_attr: str = "patient",
                    type_attr: str = "tissue",
                    df_cat: Any = None,
                    save: bool = True,
                    ) -> None:
        """Create CBIR database.

        Args:
            df_attr (Any): Pandas dataframe containing at least image files and case IDs.
            dtrs (np.ndarray, optional): pre-computed DTRs for the image files. The order should be the same as the img_attr. Default to None. 
            img_attr (str, optional): Column name of image files in df_attr. Defaults to "imgfile".
            case_attr (str, optional): Column name of case ID in df_attr. Defaults to "patient".
            type_attr (str, optional): Column name of additional attribute to show in df_attr. Defaults to "tissue".
            df_cat (Any): Pandas dataframe containing information of type_attr (e.g. differential diagnosis, text). Defaults to None.
            save (bool, optional): Saves database in the project direcotry if True. Defaults to True.
        """

        self.df_attr = df_attr
        self.img_attr = img_attr
        self.case_attr = case_attr
        self.type_attr = type_attr
        self.df_cat = df_cat

        imgfiles = df_attr[img_attr]

        if dtrs is None:
            self.dtrs = self.dtr_obj.get_dtr_multifiles(imgfiles)
        else:
            self.dtrs = dtrs
            
        #make index    
        params = {'M': 20, 'post': 0, 'efConstruction': 500}
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.addDataPointBatch(self.dtrs)
        self.index.createIndex(index_params = params)

        
        # calculated the number of cases in each category
        cases = np.array(df_attr[case_attr])
        ucases = np.unique(cases)
        types = np.array(df_attr[type_attr])
        cats = [types[np.where(cases == case)[0][0]] for case in ucases]
        self.cat_counter = collections.Counter(cats)

        if save:
            self.save_db()

    def save_db(self):
        """Saves database in the project directory.
        """
        self.index.saveIndex(filename=self.indexfile)

        joblib.dump({'img_attr':self.img_attr,
                    'case_attr':self.case_attr,
                    'type_attr':self.type_attr,
                    'cat_counter':self.cat_counter},
                    '{}/{}/attr.pkl'.format(self.working_dir,
                                            self.project,
                    ))

        self.df_attr.to_pickle('{}/{}/df.gz'.format(self.working_dir,
                                                self.project))
        if self.df_cat is not None:
            self.df_cat.to_pickle('{}/{}/df_cat.gz'.format(self.working_dir,
                                                    self.project))
        np.save('{}/{}/feats.npy'.format(self.working_dir,
                                            self.project),
                self.dtrs)

    def load_db(self):
        """Loads database from the project directory.
        """
        self.dtrs = np.load('{}/{}/feats.npy'.format(self.working_dir,
                                                        self.project))
        self.df_attr = joblib.load('{}/{}/df.gz'.format(self.working_dir,
                                                        self.project))
        try:
            self.df_cat = joblib.load('{}/{}/df_cat.gz'.format(self.working_dir,
                                                            self.project))
        except:
            self.df_cat = None
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.loadIndex(filename = self.indexfile)

        attr = joblib.load('{}/{}/attr.pkl'.format(self.working_dir,
                                            self.project))
        self.img_attr = attr['img_attr']
        self.case_attr = attr['case_attr']
        self.type_attr = attr['type_attr']
        self.cat_counter = attr['cat_counter']

        print (f"{self.project} loaded. img_attr:{self.img_attr}, case_attr:{self.case_attr}, type_attr{self.type_attr}")

    def show_db(self,
                n: int = 50,
                cases: List[str] = None,
                attrs: List[str] = None,
                ):
        """Show database images.

        Args:
            n (int, optional): The number of images shown. Defaults to 50.
            cases (List[str], optional): Cases shown. Defaults to None.
            attrs (List[str], optional): Attributes shown. Defaults to None.
        """
        df_tmp = self.df_attr
        if cases is not None:
            df_tmp = df_tmp[df_tmp[self.case_attr].isin(cases)]
        if attrs is not None:
            df_tmp = df_tmp[df_tmp[self.type_attr].isin(attrs)]

        nrow = df_tmp.shape[0]
        n = min(nrow, n)
        print (f"plot {n} images")
        df_tmp = df_tmp.sample(n=n)
        labels = ["{}\n{}\n{}".format(os.path.basename(d[1][self.img_attr]), 
                                      d[1][self.type_attr], 
                                      d[1][self.case_attr]) for d in df_tmp.iterrows()]
        imgcats(df_tmp[self.img_attr], labels=labels)
            

    def search(self,
               qimgfile: str,
               n: int = 50,
               show_query: bool = True,
               show: bool = True,
               fkey: Union[None, str] = None,
               fval: Union[str, List[str], None] = None,
               dpi: int = 320, 
               save: bool = False, 
               outfile: str = "", 
               ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Search and show images similar to the query image using DTR

        Args:
            qimgfile (str): Query image file.
            show_query (bool, optional): Show query image. Defaults to True.
            show (bool, optional): Show retrieved images. Defaults to True.
            n (int, optional): The number of retrieved images. Defaults to 50.
            scale (Union[None, int], optional): Query image is rescaled. Default to None.
            fkey (Union[None, str]): Key for filter in df_attr. Default to None.
            fval (Union[str, List[str], None]): Value(s) for filter. Default to None.
            dpi (int, optional): Dots per inch (DPI) of output image. Defaults to 320.
            save (bool, optional): Save the output image to outfile if True. Defaults to False.
            outfile (str, optional): Output image file. Defaults to "".
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Retrieved image file and the corresponding info(case, similarity, and attribute).
        """
        if type(fval) == str:
            fval = [fval]

        if fkey is not None:
            if not fkey in self.df_attr.columns:
                raise Exception("invalid key for filter {}".format(fkey))

        if save is False:
            outfile = ""
        

        qdtr = self.dtr_obj.get_dtr(qimgfile)
        qdtr_rot = self.dtr_obj.get_dtr(qimgfile, angle = 90)

        ## search
        k = min(self.df_attr.shape[0], n * 50) # the number of retrieved nearest neighbors
        
        results, dists = self._nearest_neighbor(qdtr, qdtr_rot, k)

        patients = []
        attrs = []
        sims = []
        num = []
        imgfiles = []
        for res, dist in zip(results, dists):
            imgfile = self.df_attr[self.img_attr][res]
            data = self.df_attr.iloc[res,]
            patient = data[self.case_attr]
            attr = data[self.type_attr]
            #mag = data.magnification
            if fkey is not None:
                v = self.df_attr[fkey][res]
                if not v in fval:
                    continue
                

            if not patient in patients: #remove patient-level duplicates
                patients.append(patient)
                attrs.append(attr)
                num.append(res)
                imgfiles.append(imgfile)
                sims.append(1.0 - dist)
            if len(num) == n:
                break

        ### plot results


        labels = ["{}\n{}\n{:.3f}".format(attr, patient, s) for attr,patient,s in zip(attrs, patients, sims)]
        if show:
            if show_query:
                imgcats([qimgfile, *imgfiles], labels=["query", *labels], save=outfile, dpi=dpi)
            else:
                imgcats(imgfiles, labels=labels, save=outfile, dpi=dpi)

        max_category = self._weighted_knn(sims, attrs, n=n)
        print ("The most probable diagnosis: ", max_category)

        return imgfiles, pd.DataFrame({'attr':attrs, 'case':patients, 'similarity':sims})

    def _nearest_neighbor(self,
                          qdtr: np.ndarray,
                          qdtr_rot: np.ndarray,
                          k: int,
                          ) -> Tuple[np.ndarray, np.ndarray]:

        results1, dists1 = self.index.knnQuery(qdtr, k=k)
        results2, dists2 = self.index.knnQuery(qdtr_rot, k=k)
        results = np.concatenate([results1, results2])
        dists = np.concatenate([dists1, dists2])
        s = np.argsort(dists)
        results = results[s]
        dists = dists[s]

        return results, dists

    def search_multi(self,
                    qimgfiles: List[str],
                    strategy: str = 'max',
                    n: int = 50,
                    show_query: bool = True,
                    show: bool = True,
                    fkey: Union[None, str] = None,
                    fval: Union[str, List[str], None] = None,
                    dpi: int = 320,
                    save: bool = False,
                    outfile: str = "", 
                    ) -> pd.DataFrame:
        """Search and show images similar to the query image using DTR

        Args:
            qimgfiles (List[str]): Query image files.
            strategy (str, optional): Search strategy. In 'max', the similarity for the case is cacluated based on the maximum similarity among queries. In 'mean', the similarity for the case is cacluated based on the average similarity among queries. 
            show_query (bool, optional): Show query images. Defaults to True.
            show (bool, optional): Show retrieved images. Defaults to True.
            n (int, optional): The number of retrieved images. Defaults to 50.
            scale (Union[None, int], optional): Query images are rescaled. Default to None.
            fkey (Union[None, str]): Key for filter in df_attr. Default to None.
            fval (Union[str, List[str], None]): Value(s) for filter. Default to None.
            dpi (int, optional): Dots per inch (DPI) of output image. Defaults to 320.
            save (bool, optional): Save the output image to outfile if True. Defaults to False.
            outfile (str, optional): Output image file. Defaults to "".
        Returns:
            pd.DataFrame : Results
        """
        if type(fval) == str:
            fval = [fval]

        if fkey is not None:
            if not fkey in self.df_attr.columns:
                raise Exception("invalid key for filter {}".format(fkey))
        
        if save is False:
            outfile = ""
        
        if not strategy in ['max', 'mean']:
            raise Exception(f'invalid strategy: {strategy}')


        df_each = []
        for i, qimgfile in enumerate(qimgfiles):
            qdtr = self.dtr_obj.get_dtr(qimgfile)
            qdtr_rot = self.dtr_obj.get_dtr(qimgfile, angle = 90)

            ## search
            k = min(self.df_attr.shape[0], n * 50) # the number of retrieved nearest neighbors

            results, dists = self._nearest_neighbor(qdtr, qdtr_rot, k)

            patients = []
            sims = []
            cats = []
            num = []
            imgfiles = []
            for res, dist in zip(results, dists):
                imgfile = self.df_attr[self.img_attr][res]
                data = self.df_attr.iloc[res,]
                patient = data[self.case_attr]
                category = data[self.type_attr]

                if fkey is not None:
                    v = self.df_attr[fkey][res]
                    if not v in fval:
                        continue

                if not patient in patients: #remove patient-level duplicates
                    patients.append(patient)
                    num.append(res)
                    imgfiles.append(imgfile)
                    sims.append(1.0 - dist)
                    cats.append(category)
            df_each.append(pd.DataFrame({f'patient':patients, f'num_{i}':num, 
                                         f'sim_{i}':sims, f'imgfile_{i}':imgfiles,
                                        f'category_{i}':cats}))
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=['patient'], how='outer'), df_each)
        df_merged = df_merged.fillna(0)
        
        df_merged['agg_sim'] = df_merged.filter(like='sim_').agg(strategy, axis=1)
        df_merged = df_merged.sort_values('agg_sim', ascending=False)
        df_merged = df_merged.iloc[:min(n, df_merged.shape[0]),:]

        max_category = self._weighted_knn(df_merged['agg_sim'], df_merged['category_0'], n=n)
        print ("The most probable diagnosis: ", max_category)

        qn = len(qimgfiles)
      
        if show:
            nrows = qn
            ncols = df_merged.shape[0]
            if show_query:
                ncols += 1
                offset = qn + 1
                for j, qimgfile in enumerate(qimgfiles):
                    plt.subplot(ncols, nrows, j + 1) 
                    im_list = np.asarray(Image.open(qimgfile))
                    plt.imshow(im_list)
                    plt.axis('off')
                    plt.title('query: {}'.format(qimgfile))
            else:
                offset = 1
                
            for i, d in enumerate(df_merged.iterrows()):
                for j in range(qn):
                    plt.subplot(ncols, nrows, j + i * qn + offset)
                    imgfile = d[1][f'imgfile_{j}']
                    if type(imgfile) != int:
                        if os.path.exists(imgfile):
                            im_list = np.asarray(Image.open(imgfile)) 
                            plt.imshow(im_list)
                            plt.title('sim:{:.3f}'.format(d[1][f'sim_{j}']))
                    plt.axis('off')
            if save is True:
                plt.savefig(outfile, dpi=dpi)
            
        return df_merged

        
    def _weighted_knn(self, 
                      sims, 
                      cats, 
                      n: int = 10):
        """Distance and 1/N_samples weighted kNN

        Args:
            sims (np.ndarray): similarity.
            cats (list): categories.
            n (int, optional): The number of retrieved images. Defaults to 10.
        """
        weights = np.array([1./(1.01-s)/self.cat_counter[c] for s,c in zip(sims, cats)])
        weights = weights[:min(n, len(sims))]
        df = pd.DataFrame({'category':cats,
                           'weight': weights})
        max_category = df.groupby('category').sum().idxmax().values[0]
        return max_category

