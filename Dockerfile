# A image for building paddle binaries
# Use cuda devel base image for both cpu and gpu environment
# When you modify it, please be aware of cudnn-runtime version
# and libcudnn.so.x in paddle/scripts/docker/build.sh
FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

ARG UBUNTU_MIRROR
RUN /bin/bash -c 'if [[ -n ${UBUNTU_MIRROR} ]]; then sed -i 's#http://archive.ubuntu.com/ubuntu#${UBUNTU_MIRROR}#g' /etc/apt/sources.list; fi'

# ENV variables
ARG WITH_GPU
ARG WITH_AVX
ARG WITH_DOC

ENV WOBOQ OFF
ENV WITH_GPU=${WITH_GPU:-ON}
ENV WITH_AVX=${WITH_AVX:-ON}
ENV WITH_DOC=${WITH_DOC:-OFF}

ENV HOME /root
# Add bash enhancements
COPY ./paddle/scripts/docker/root/ /root/

RUN apt-get update && \
    apt-get install -y --allow-downgrades \
    git python-pip python-dev python-opencv openssh-server bison \
    libnccl2=2.1.2-1+cuda8.0 libnccl-dev=2.1.2-1+cuda8.0 \
    wget unzip unrar tar xz-utils bzip2 gzip coreutils ntp \
    curl sed grep graphviz libjpeg-dev zlib1g-dev  \
    python-matplotlib gcc-4.8 g++-4.8 \
    automake locales clang-format swig cmake  \
    liblapack-dev liblapacke-dev \
    clang-3.8 llvm-3.8 libclang-3.8-dev \
    net-tools libtool ccache && \
    apt-get clean -y

# Install Go and glide
RUN wget -qO- https://storage.googleapis.com/golang/go1.8.1.linux-amd64.tar.gz | \
    tar -xz -C /usr/local && \
    mkdir /root/gopath && \
    mkdir /root/gopath/bin && \
    mkdir /root/gopath/src
ENV GOROOT=/usr/local/go GOPATH=/root/gopath
# should not be in the same line with GOROOT definition, otherwise docker build could not find GOROOT.
ENV PATH=${PATH}:${GOROOT}/bin:${GOPATH}/bin
# install glide
RUN curl -s -q https://glide.sh/get | sh

# Install TensorRT
# following TensorRT.tar.gz is not the default official one, we do two miny changes:
# 1. Remove the unnecessary files to make the library small. TensorRT.tar.gz only contains include and lib now,
#    and its size is only one-third of the official one.
# 2. Manually add ~IPluginFactory() in IPluginFactory class of NvInfer.h, otherwise, it couldn't work in paddle.
#    See https://github.com/PaddlePaddle/Paddle/issues/10129 for details.
RUN wget -qO- http://paddlepaddledeps.cdn.bcebos.com/TensorRT-4.0.0.3.Ubuntu-16.04.4.x86_64-gnu.cuda-8.0.cudnn7.0.tar.gz | \
    tar -xz -C /usr/local && \
    cp -rf /usr/local/TensorRT/include /usr && \
    cp -rf /usr/local/TensorRT/lib /usr

# git credential to skip password typing
RUN git config --global credential.helper store

# Fix locales to en_US.UTF-8
RUN localedef -i en_US -f UTF-8 en_US.UTF-8

# FIXME: due to temporary ipykernel dependency issue, specify ipykernel jupyter
# version util jupyter fixes this issue.

# specify sphinx version as 1.5.6 and remove -U option for [pip install -U
# sphinx-rtd-theme] since -U option will cause sphinx being updated to newest
# version(1.7.1 for now), which causes building documentation failed.
RUN easy_install -U pip && \
    pip install -U wheel && \
    pip install -U docopt PyYAML sphinx==1.5.6 && \
    pip install sphinx-rtd-theme==0.1.9 recommonmark

RUN pip install pre-commit 'ipython==5.3.0' && \
    pip install 'ipykernel==4.6.0' 'jupyter==1.0.0' && \
    pip install opencv-python

#For docstring checker
RUN pip install pylint pytest astroid isort

COPY ./python/requirements.txt /root/
RUN pip install -r /root/requirements.txt

# To fix https://github.com/PaddlePaddle/Paddle/issues/1954, we use
# the solution in https://urllib3.readthedocs.io/en/latest/user-guide.html#ssl-py2
RUN apt-get install -y libssl-dev libffi-dev
RUN pip install certifi urllib3[secure]


# Install woboq_codebrowser to /woboq
RUN git clone https://github.com/woboq/woboq_codebrowser /woboq && \
    (cd /woboq \
     cmake -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-3.8 \
           -DCMAKE_BUILD_TYPE=Release . \
     make)

# Configure OpenSSH server. c.f. https://docs.docker.com/engine/examples/running_ssh_service
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
EXPOSE 22
