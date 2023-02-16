FROM nvcr.io/nvidia/pytorch:21.07-py3

LABEL maintainer="D.Komura <kdais-prm@m.u-tokyo.ac.jp>" \
    version="0.2.5_clip"

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install \
    nmslib \
    seaborn \
    umap-learn \
    ftfy \
    plotly

RUN pip install \
    git+https://github.com/openai/CLIP.git
RUN pip install \
    git+https://github.com/dakomura/deep_texture_histology.git@clip

RUN pip install cvxopt
RUN pip install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm

RUN git clone -b clip https://github.com/dakomura/deep_texture_histology.git
