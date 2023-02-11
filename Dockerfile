FROM pytorch/pytorch


# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]


RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
        wget \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

ARG PYTHON=python3
ARG PIP=pip3


ENV LANG C.UTF-8


RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip


# Create the environment:
COPY environment.yml .
RUN conda env update --file  environment.yml --prune

RUN ${PIP} --no-cache-dir install --upgrade  pip
RUN ${PIP} --no-cache-dir install setuptools 
RUN ${PIP} --no-cache-dir install hdf5storage 
RUN ${PIP} --no-cache-dir install h5py 
RUN ${PIP} --no-cache-dir install py3nvml 
RUN ${PIP} --no-cache-dir install scikit-image 
RUN ${PIP} --no-cache-dir install scikit-learn 
RUN ${PIP} --no-cache-dir install matplotlib  
RUN ${PIP} --no-cache-dir install rdkit-pypi 
RUN ${PIP} --no-cache-dir install tensorflow-gpu 
RUN ${PIP} --no-cache-dir install nfp 
RUN ${PIP} --no-cache-dir install torch-geometric 
RUN ${PIP} --no-cache-dir install torch-scatter  
RUN ${PIP} --no-cache-dir install torch-sparse 
RUN ${PIP} --no-cache-dir install torch-cluster 
RUN ${PIP} --no-cache-dir install ogb 
RUN ${PIP} --no-cache-dir install tqdm 
# https://www.dgl.ai/pages/start.html
RUN ${PIP} install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
