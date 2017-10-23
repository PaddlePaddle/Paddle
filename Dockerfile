# A image for building paddle binaries
# Use cuda devel base image for both cpu and gpu environment
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
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
    apt-get install -y \
    git python-pip python-dev openssh-server bison libnccl-dev \
    wget unzip unrar tar xz-utils bzip2 gzip coreutils ntp \
    curl sed grep graphviz libjpeg-dev zlib1g-dev  \
    python-matplotlib gcc-4.8 g++-4.8 \
    automake locales clang-format swig doxygen cmake  \
    liblapack-dev liblapacke-dev libboost-dev \
    clang-3.8 llvm-3.8 libclang-3.8-dev \
    net-tools && \
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

# git credential to skip password typing
RUN git config --global credential.helper store

# Fix locales to en_US.UTF-8
RUN localedef -i en_US -f UTF-8 en_US.UTF-8

# FIXME: due to temporary ipykernel dependency issue, specify ipykernel jupyter
# version util jupyter fixes this issue.
RUN pip install --upgrade pip && \
    pip install -U wheel && \
    pip install -U docopt PyYAML sphinx && \
    pip install -U sphinx-rtd-theme==0.1.9 recommonmark

RUN pip install pre-commit 'ipython==5.3.0' && \
    pip install 'ipykernel==4.6.0' 'jupyter==1.0.0' && \
    pip install opencv-python

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

# development image default do build work
CMD ["bash", "/paddle/paddle/scripts/docker/build.sh"]
