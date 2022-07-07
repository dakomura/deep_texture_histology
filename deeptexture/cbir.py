from functools import reduce
import os
from typing import Any, List
import nmslib
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from .dtr import *
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
                    img_attr: str = "imgfile",
                    case_attr: str = "patient",
                    type_attr: str = "tissue",
                    save: bool = True,
                    ) -> None:
        """Create CBIR database.

        Args:
            df_attr (Any): Pandas dataframe containing at least image files and case IDs.
            img_attr (str, optional): Column name of image files in df_attr. Defaults to "imgfile".
            case_attr (str, optional): Column name of case ID in df_attr. Defaults to "patient".
            type_attr (str, optional): Column name of additional attribute to show in df_attr. Defaults to "tissue".
            save (bool, optional): Saves database in the project direcotry if True. Defaults to True.
        """

        self.df_attr = df_attr
        self.img_attr = img_attr
        self.case_attr = case_attr
        self.type_attr = type_attr

        imgfiles = df_attr[img_attr]

        self.dtrs = self.dtr_obj.get_dtr_multifiles(imgfiles)
            
        #make index    
        params = {'M': 20, 'post': 0, 'efConstruction': 500}
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.addDataPointBatch(self.dtrs)
        self.index.createIndex(index_params = params)

        if save:
            self.save_db()

    def save_db(self):
        """Saves database in the project directory.
        """
        self.index.saveIndex(filename=self.indexfile)

        joblib.dump({'img_attr':self.img_attr,
                    'case_attr':self.case_attr,
                    'type_attr':self.type_attr},
                    '{}/{}/attr.pkl'.format(self.working_dir,
                                            self.project,
                    ))

        self.df_attr.to_pickle('{}/{}/df.gz'.format(self.working_dir,
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
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.loadIndex(filename = self.indexfile)

        attr = joblib.load('{}/{}/attr.pkl'.format(self.working_dir,
                                            self.project))
        self.img_attr = attr['img_attr']
        self.case_attr = attr['case_attr']
        self.type_attr = attr['type_attr']

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
        self._imgcats(df_tmp[self.img_attr], labels=labels)
            

    def search(self,
               qimgfile: str,
               n: int = 50,
               show_query: bool = True,
               show: bool = True,
               scale: Union[None, int] = None,
               fkey: Union[None, str] = None,
               fval: Union[str, List[str], None] = None,
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
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Retrieved image file and the corresponding info(case, similarity, and attribute).
        """
        if type(filter_val) == str:
            filter_val = [filter_val]

        if fkey is not None:
            if not fkey in self.df_attr.columns:
                raise Exception("invalid key for filter {}".format(fkey))
        

        qdtr = self.dtr_obj.get_dtr(qimgfile, scale=scale)
        qdtr_rot = self.dtr_obj.get_dtr(qimgfile, angle = 90, scale=scale)

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
                self._imgcats([qimgfile, *imgfiles], labels=["query", *labels])
            else:
                self._imgcats(imgfiles, labels=labels)

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
                    scale: Union[None, int] = None,
                    fkey: Union[None, str] = None,
                    fval: Union[str, List[str], None] = None,
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
        Returns:
            pd.DataFrame : Results
        """
        if type(filter_val) == str:
            filter_val = [filter_val]

        if fkey is not None:
            if not fkey in self.df_attr.columns:
                raise Exception("invalid key for filter {}".format(fkey))
        
        if not strategy in ['max', 'mean']:
            raise Exception(f'invalid strategy: {strategy}')


        df_each = []
        for i, qimgfile in enumerate(qimgfiles):
            qdtr = self.dtr_obj.get_dtr(qimgfile, scale=scale)
            qdtr_rot = self.dtr_obj.get_dtr(qimgfile, angle = 90, scale=scale)

            ## search
            k = min(self.df_attr.shape[0], n * 50) # the number of retrieved nearest neighbors

            results, dists = self._nearest_neighbor(qdtr, qdtr_rot, k)

            patients = []
            sims = []
            num = []
            imgfiles = []
            for res, dist in zip(results, dists):
                imgfile = self.df_attr[self.img_attr][res]
                data = self.df_attr.iloc[res,]
                patient = data[self.case_attr]

                if fkey is not None:
                    v = self.df_attr[fkey][res]
                    if not v in fval:
                        continue

                if not patient in patients: #remove patient-level duplicates
                    patients.append(patient)
                    num.append(res)
                    imgfiles.append(imgfile)
                    sims.append(1.0 - dist)
            df_each.append(pd.DataFrame({f'patient':patients, f'num_{i}':num, 
                                         f'sim_{i}':sims, f'imgfile_{i}':imgfiles}))
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=['patient'], how='outer'), df_each)
        df_merged = df_merged.fillna(0)
        
        df_merged['agg_sim'] = df_merged.filter(like='sim_').agg(strategy, axis=1)
        df_merged = df_merged.sort_values('agg_sim', ascending=False)
        df_merged = df_merged.iloc[:min(n, df_merged.shape[0]),:]

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
                    if os.path.exists(imgfile):
                        im_list = np.asarray(Image.open(imgfile)) 
                        plt.imshow(im_list)
                        plt.axis('off')
            
        return df_merged

    def _imgcats(self, 
                    infiles: List[str], 
                    labels: List[str], 
                    nrows: int = 3, 
                    ) -> None:

        ncols = int(np.ceil(len(infiles)/nrows))
        for i, infile in enumerate(infiles):
            plt.subplot(ncols, nrows, i+1)
            im = Image.open(infile)
            im_list = np.asarray(im)
            plt.imshow(im_list)
            if len(labels) != 0:
                plt.title(labels[i])
            plt.axis('off')
        plt.show()