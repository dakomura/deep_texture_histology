FROM nvcr.io/nvidia/pytorch:21.07-py3
# or select appropriate image for your environment: https://hub.docker.com/r/tensorflow/tensorflow/tags
# to build, run `docker build . -t dkomura/deeptexture:0.2.2`
# to run container, run `docker run -it --gpus 1 --net=host dkomura/deeptexture:0.2.2`

LABEL maintainer="D.Komura <kdais-prm@m.u-tokyo.ac.jp>" \
    version="0.2.5_clip"

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y tzdata && apt-get install -y nodejs npm

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

RUN git clone -b clip https://github.com/dakomura/deep_texture_histology.git

RUN jupyter labextension install jupyterlab-plotly@4.8.2
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.8.2

CMD ["jupyter", "lab", "--allow-root", "--ip", "0.0.0.0"]
