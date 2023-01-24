import platform
import numpy
import tensorflow
import joblib
import PIL
import nmslib
import matplotlib
import sklearn
import seaborn
import pandas
import cv2

'''
[ Prerequisites ]
Python version 3.6 or newer.
numpy >=1.20.3
tensorflow >=2.1.0
joblib >=0.13.2
Pillow >=8.0.1
nmslib >=2.0.6
matplotlib >= 3.5.0
scikit-learn >=1.0.2
seaborn >=0.10.1
pandas >=1.1.0
cv
'''

def compare_version(lib1, lib2):
    # Minimum required version
    list1 = [int(s) for s in lib1.split('.')]
    # Installed version
    list2 = [int(s) for s in lib2.split('.')]

    if len(list1) == len(list2):
        for i in range(len(list1)):
            if list1[i] == list2[i]:
                if i != len(list1) - 1:
                    # print('next value check...')
                    pass
                else:
                    print('{} is the same version as the required version\n'.format(lib2))
                    return 'same'
                pass

            elif list1[i] > list2[i]:
                print('{} is older than the required version\n'.format(lib2))
                print(list1[i])
                return 'older'
            else:
                print('{} is newer than the required version\n'.format(lib2))
                return 'newer'
    else:
        print('can not compare versions')
        return 'false'

python_min_ver = '3.6.0'
numpy_min_ver = '1.20.3'
tensorflow_min_ver = '2.1.0'
joblib_min_ver = '0.13.2'
pillow_min_ver = '8.0.1'
nmslib_min_ver = '2.0.6'
matplotlib_min_ver = '3.5.0'
# sklearn_min_ver = '1.1.0'
sklearn_min_ver = '1.0.2'
seaborn_min_ver = '0.10.1'
pandas_min_ver = '1.1.0'
opencv_min_ver = '0.0.0'

print('\nStart Quick check\n')

version_check_results = []
print('Python version: ', platform.python_version())
version_check_results.append(compare_version(python_min_ver, platform.python_version()))
print('Numpy version: ', numpy.__version__)
version_check_results.append(compare_version(numpy_min_ver, numpy.__version__))
print('Tensorflow version: ', tensorflow.__version__)
version_check_results.append(compare_version(tensorflow_min_ver, tensorflow.__version__))
print('Joblib version: ', joblib.__version__)
version_check_results.append(compare_version(joblib_min_ver, joblib.__version__))
print('Pillow version: ', PIL.__version__)
version_check_results.append(compare_version(pillow_min_ver, PIL.__version__))
print('Nmslib version: ', nmslib.__version__)
version_check_results.append(compare_version(nmslib_min_ver, nmslib.__version__))
print('Matplotlib version: ', matplotlib.__version__)
version_check_results.append(compare_version(matplotlib_min_ver, matplotlib.__version__))
print('Scikit-learn version: ', sklearn.__version__)
version_check_results.append(compare_version(sklearn_min_ver, sklearn.__version__))
print('Seaborn version: ', seaborn.__version__)
version_check_results.append(compare_version(seaborn_min_ver, seaborn.__version__))
print('Pandas version: ', pandas.__version__)
version_check_results.append(compare_version(pandas_min_ver, pandas.__version__))
print('OpenCV version: ', cv2.__version__)
version_check_results.append(compare_version(opencv_min_ver, cv2.__version__))

print('version_check_results: {}\n'.format(version_check_results))

try:
    for result in version_check_results:
        if result == 'older' or result == 'false':
            raise Exception
        else:
            pass
except Exception:
    print('Please check libraries versions\n')
    exit()

import deeptexture as dt
import glob
from PIL import Image
numpy.set_printoptions(threshold=1024, suppress=True)

print('Deeptexture version: {}\n'.format(dt.__version__))

dtr_obj = dt.DTR(arch='vgg', layer='block4_conv3', dim=1024)

imgfile_list = glob.glob('./example/*.jpg')
# imgfile = './example/0KNpXTsaNix2T6.jpg'
imgfile = imgfile_list[0]

print('\n')
print('# calculate DTR for one image file')
dtr = dtr_obj.get_dtr(imgfile)
print('Result: \n{}\n'.format(dtr))

print('# calculate mean DTR for unrotated and rotated image file')
dtr_rot = dtr_obj.get_dtr(imgfile, angle=[0, 90])
print('Result: \n{}\n'.format(dtr_rot))

print('# calculate DTR for one image object')
img = Image.open(imgfile)
dtr_from_img_obj = dtr_obj.get_dtr(img)
print('Result: \n{}\n'.format(dtr_from_img_obj))

print('End Quick check\n')

