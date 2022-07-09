FROM tensorflow/tensorflow:2.6.1-gpu-jupyter
# or select appropriate image for your environment: https://hub.docker.com/r/tensorflow/tensorflow/tags
# to build, run `docker build . -t dkomura/deeptexture:0.2.2`
# to run container, run `docker run -it --gpus 1 --net=host dkomura/deeptexture:0.2.2`

LABEL maintainer="D.Komura <kdais-prm@m.u-tokyo.ac.jp>" \
    version="0.2.2"

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

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

RUN github clone https://github.com/dakomura/deep_texture_histology.git

CMD ["jupyter", "lab", "--allow-root", "--ip", "0.0.0.0"]