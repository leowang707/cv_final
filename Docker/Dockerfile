FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

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

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

RUN pip --no-cache-dir install Cython

COPY requirements.txt /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt

RUN pip install --upgrade pycocotools

RUN pip install jupyter seaborn

##################################### OpenCV ###################################

# Install additional libraries required for OpenCV
RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install OpenCV Python package
RUN pip install opencv-python-headless  # Use opencv-python-headless if you don't need GUI functionalities

##################################### Detectron2 ###################################

# Install detectron2 dependencies
RUN pip install -U torch torchvision torchaudio
RUN pip install cython pyyaml matplotlib tqdm tensorboard cloudpickle

# Clone detectron2 from GitHub
RUN rm -rf /detectron2 && git clone https://github.com/facebookresearch/detectron2.git /detectron2

# Install detectron2
WORKDIR /detectron2

# Run setup.py to install detectron2
RUN python setup.py build develop

##################################### setting ###################################################

RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "${USER} ALL=(ALL)  ALL" >> /etc/sudoers

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

USER ${USER}
WORKDIR ${HOME}
CMD ["/bin/bash"]
