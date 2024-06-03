FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=leowang707 \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:111111" | chpasswd

##################################### PIP ######################################

# Update the package list, perform a distribution upgrade, and install essential packages
RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
    && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \ 
    locales \
    cmake \
    unzip \
    make \
    git \
    vim \
    gedit \
    wget \
    sudo \
    lsb-release \
    build-essential \
    net-tools \
    dirmngr \
    gnupg2 \
    mercurial \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libpcap-dev \
    python3-pip \
    python3-setuptools \
    python3-opencv \
    python3-numpy \
    python3-empy  \
    python3-dev \ 
    python3-tk \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --upgrade pip setuptools

RUN pip3 install \
    matplotlib \
    pandas \
    quart \
    rich \
    seaborn \
    ftfy \
    opencv-python

# Install PyTorch, torchvision, and torchaudio with CUDA 11.8 support
RUN pip3 install \
    https://download.pytorch.org/whl/cu118/torch-2.3.0%2Bcu118-cp38-cp38-linux_x86_64.whl#sha256=d4375df5c2ea5996578faddece62cd31621719686904086ec4242d40cb37247f \  
    https://download.pytorch.org/whl/cu118/torchvision-0.18.0%2Bcu118-cp38-cp38-linux_x86_64.whl#sha256=8ad3a67e0484f147bd5e6741a597e4a33416487f163fdcf3eed31431d4c84da7 \
    https://download.pytorch.org/whl/cu118/torchaudio-2.3.0%2Bcu118-cp38-cp38-linux_x86_64.whl#sha256=c92ecea52a9f239429619ca6172896c6d2e0323c9cd6f4edd08363be2e82412d    

RUN pip install 'git+https://github.com/cocodataset/panopticapi.git'

RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

RUN pip3 install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html

RUN pip3 install scikit-learn

RUN pip3 install transformers \
    datasets \
    evaluate \
    transformers[sentencepiece] \
    accelerate \
    timm \
    albumentations

RUN pip3 install jupyter 

##################################### setting ###################################################

RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "${USER} ALL=(ALL)  ALL" >> /etc/sudoers

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

USER ${USER}
WORKDIR ${HOME}
CMD ["/bin/bash"]
