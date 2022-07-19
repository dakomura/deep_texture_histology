FROM tensorflow/tensorflow:2.6.1-gpu-jupyter
# or select appropriate image for your environment: https://hub.docker.com/r/tensorflow/tensorflow/tags

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install \
    joblib \
    nmslib \
    scikit-learn \
    seaborn \
    pandas \
    jupyterlab \
    umap-learn \
    opencv-python \
    plotly \
    deeptexture