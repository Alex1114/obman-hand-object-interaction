FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO melodic

###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=arg-medical \
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

###################################### basic tools #####################################

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
    && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \
    locales \
    cmake \
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
    python-gtk2 \
    python-gobject \
    python-tk \
    python-pip \
    python3-pip \
    python3-setuptools \
    python3-opencv \
    python3-numpy \
    python3-empy  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*



##################################### PIP2 ######################################

RUN pip2 install --upgrade pip setuptools

RUN pip2 install \
    modern_robotics 

##################################### PIP3 ######################################

RUN pip3 install --upgrade pip setuptools

RUN pip3 install \
    digi-xbee==1.3.0 \
    pyyaml \
    rospkg \
    cython \
    catkin-tools \
    matplotlib \
    pandas \
    pypozyx \
    requests \
    scikit-image \
    scikit-learn \
    gdown \
    seaborn \
    modern_robotics \ 
    chumpy \ 
    trimesh \
    pycpd \
    easydict \
    shapely \
    lmdb \ 
    colour \ 
    pyrr \
    plotly \
    progress \ 
    pybullet \
    rtree \
    sk-video

RUN pip3 install \
    torch==1.1.0 \
    torchvision==0.3.0 \
    -f https://download.pytorch.org/whl/torch_stable.html


##################################### setting #####################################
RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "arg-medical ALL=(ALL)  ALL" > /etc/sudoers

# Support of nvidia-docker 2.0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Pytorch model
COPY ./resnet18-5c106cde.pth /home/arg-medical/.cache/torch/checkpoints/

USER ${USER}

WORKDIR ${HOME}
