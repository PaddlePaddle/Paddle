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
ARG WITH_STYLE_CHECK

ENV WOBOQ OFF
ENV WITH_GPU=${WITH_GPU:-OFF}
ENV WITH_AVX=${WITH_AVX:-ON}
ENV WITH_DOC=${WITH_DOC:-OFF}
ENV WITH_STYLE_CHECK=${WITH_STYLE_CHECK:-OFF}

ENV HOME /root
# Add bash enhancements
COPY ./paddle/scripts/docker/root/ /root/

RUN apt-get update && \
    apt-get install -y \
    git python-pip python-dev openssh-server bison  \
    wget unzip tar xz-utils bzip2 gzip coreutils ntp \
    curl sed grep graphviz libjpeg-dev zlib1g-dev  \
    python-numpy python-matplotlib gcc g++ \
    automake locales clang-format-3.8 swig doxygen cmake  \
    liblapack-dev liblapacke-dev libboost-dev \
    clang-3.8 llvm-3.8 libclang-3.8-dev \
    net-tools && \
    apt-get clean -y

# Install Go
RUN wget -O go.tgz https://storage.googleapis.com/golang/go1.8.1.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go.tgz && \
    mkdir /root/gopath && \
    rm go.tgz
ENV GOROOT=/usr/local/go GOPATH=/root/gopath
# should not be in the same line with GOROOT definition, otherwise docker build could not find GOROOT.
ENV PATH=${PATH}:${GOROOT}/bin

# git credential to skip password typing
RUN git config --global credential.helper store

# Fix locales to en_US.UTF-8
RUN localedef -i en_US -f UTF-8 en_US.UTF-8

# FIXME: due to temporary ipykernel dependency issue, specify ipykernel jupyter
# version util jupyter fixes this issue.
RUN pip install --upgrade pip && \
    pip install -U 'protobuf==3.1.0' && \
    pip install -U wheel pillow BeautifulSoup && \
    pip install -U docopt PyYAML sphinx && \
    pip install -U sphinx-rtd-theme==0.1.9 recommonmark && \
    pip install pre-commit 'requests==2.9.2' 'ipython==5.3.0' && \
    pip install 'ipykernel==4.6.0' 'jupyter==1.0.0' && \ 
    pip install rarfile

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
