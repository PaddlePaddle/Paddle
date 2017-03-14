# A image for building paddle binaries
# Use cuda devel base image for both cpu and gpu environment
FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_MIRROR
RUN /bin/bash -c 'if [[ -n ${UBUNTU_MIRROR} ]]; then sed -i 's#http://archive.ubuntu.com#${UBUNTU_MIRROR}#g' /etc/apt/sources.list; fi'

# ENV variables
ARG BUILD_WOBOQ
ARG BUILD_AND_INSTALL
ARG WITH_AVX
ARG WITH_DOC
ARG WITH_STYLE_CHECK

ENV BUILD_WOBOQ=${BUILD_WOBOQ:-OFF}
ENV BUILD_AND_INSTALL=${BUILD_AND_INSTALL:-OFF}
ENV WITH_GPU=OFF
ENV WITH_AVX=${WITH_AVX:-ON}
ENV WITH_DOC=${WITH_DOC:-OFF}
ENV WITH_STYLE_CHECK=${WITH_STYLE_CHECK:-OFF}

ENV HOME /root

RUN apt-get update && \
    apt-get install -y git python-pip python-dev openssh-server bison && \
    apt-get install -y wget unzip tar xz-utils bzip2 gzip coreutils && \
    apt-get install -y curl sed grep graphviz libjpeg-dev zlib1g-dev && \
    apt-get install -y python-numpy python-matplotlib gcc g++ gfortran && \
    apt-get install -y automake locales clang-format-3.8 && \
    apt-get clean -y

# git credential to skip password typing
RUN git config --global credential.helper store

# Fix locales to en_US.UTF-8
RUN localedef -i en_US -f UTF-8 en_US.UTF-8

RUN pip install --upgrade pip && \
    pip install -U 'protobuf==3.1.0' && \
    pip install -U wheel pillow BeautifulSoup && \
    pip install -U docopt PyYAML sphinx && \
    pip install -U sphinx-rtd-theme==0.1.9 recommonmark && \
    pip install -U pre-commit 'requests==2.9.2' jupyter

RUN curl -sSL https://cmake.org/files/v3.4/cmake-3.4.1.tar.gz | tar -xz && \
    cd cmake-3.4.1 && ./bootstrap && make -j `nproc` && make install && \
    cd .. && rm -rf cmake-3.4.1

RUN apt-get install -y swig
# FIXME: wait a long time is OK
CMD ["sleep", "3600"]
