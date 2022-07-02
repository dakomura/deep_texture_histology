import os
from typing import Any, List
import nmslib
import joblib
import numpy as np
from .dtr import *
import matplotlib.pyplot as plt
from PIL import Image
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
        """_summary_

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
        labels = ["{}\n{}\n{}".format(os.path.basename(d[self.img_attr]), 
                                      d[self.type_attr], 
                                      d[self.case_attr]) for d in df_tmp.iterrows()]
        self._imgcats(df_tmp[self.img_attr], labels=labels)
            

    def search(self,
               qimgfile: str,
               n: int = 50,
               show_query: bool = True,
               scale: Union[None, int] = None,
               ) -> Tuple[np.ndarray, List[str]]:
        """Search and show images similar to the query image using DTR

        Args:
            qimgfile (str): Query image file.
            show_query (bool, optional): Show query image. Defaults to True.
            n (int, optional): The number of images shown. Defaults to 50.
            scale (Union[None, int], optional): Query image is rescaled. Default to None.
        Returns:
            Tuple[np.ndarray, List[str]]: Retrieved image file and the corresponding labels.
        """
        

        qdtr = self.dtr_obj.get_dtr(qimgfile, scale=scale)
        qdtr_rot = self.dtr_obj.get_dtr(qimgfile, angle = 90, scale=scale)

        ## search
        k = min(self.df_attr.shape[0], n * 50) # the number of retrieved nearest neighbors

        results1, dists1 = self.index.knnQuery(qdtr, k=k)
        results2, dists2 = self.index.knnQuery(qdtr_rot, k=k)
        results = np.concatenate([results1, results2])
        dists = np.concatenate([dists1, dists2])
        s = np.argsort(dists)
        results = results[s]
        dists = dists[s]

        patients = []
        attrs = []
        dist_list = []
        num = []
        imgfiles = []
        for res, dist in zip(results, dists):
            imgfile = self.df_attr[self.img_attr][res]
            data = self.df_attr.iloc[res,]
            patient = data[self.case_attr]
            attr = data[self.type_attr]
            #mag = data.magnification

            if not patient in patients: #remove patient-level duplicates
                patients.append(patient)
                attrs.append(attr)
                num.append(res)
                imgfiles.append(imgfile)
                dist_list.append(dist)
            if len(num) == n:
                break

        ### plot results


        labels = ["{}\n{}\n{}".format(attr, patient, 1-d) for attr,patient,d in zip(attrs, patients, dist_list)]
        if show_query:
            self._imgcats([qimgfile, *imgfiles], labels=["query", *labels])
        else:
            self._imgcats(imgfiles, labels=labels)

        return imgfiles, labels

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