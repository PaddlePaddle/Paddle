# A image for building paddle binaries
# Use cuda devel base image for both cpu and gpu environment
# When you modify it, please be aware of cudnn-runtime version
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

# ENV variables
ARG WITH_GPU
ARG WITH_AVX

ENV WITH_GPU=${WITH_GPU:-ON}
ENV WITH_AVX=${WITH_AVX:-ON}
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME /root
# Add bash enhancements
RUN apt-get update && \
    apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y curl wget vim git unzip unrar tar xz-utils bzip2 gzip \
        coreutils ntp language-pack-zh-hans python-qt4 libsm6 libxext6 libxrender-dev


# Downgrade gcc&&g++
WORKDIR /usr/bin
RUN apt-get update --fix-missing
COPY script_build /script_build
RUN bash /script_build/install_gcc.sh gcc82 && rm -rf /script_build && \
    cp gcc gcc.bak && cp g++ g++.bak && rm gcc && rm g++ && \
    ln -s /usr/local/gcc-8.2/bin/gcc /usr/local/bin/gcc && \
    ln -s /usr/local/gcc-8.2/bin/g++ /usr/local/bin/g++ && \
    ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/gcc && \
    ln -s /usr/local/gcc-8.2/bin/g++ /usr/bin/g++
ENV PATH=/usr/local/gcc-8.2/bin:$PATH

RUN apt-get update && \
    apt-get install -y python3.6 python3.6-dev python3.6-venv && \
    apt-get install -y python3-pip


# install cmake
WORKDIR /home
RUN wget -q https://cmake.org/files/v3.20/cmake-3.20.0-linux-x86_64.tar.gz && tar -zxvf cmake-3.20.0-linux-x86_64.tar.gz && rm cmake-3.20.0-linux-x86_64.tar.gz
ENV PATH=/home/cmake-3.20.0-linux-x86_64/bin:$PATH

# remove them when apt-get support 2.27 and higher version
RUN wget -q https://ftp.gnu.org/gnu/binutils/binutils-2.33.1.tar.gz && \
    tar -xzf binutils-2.33.1.tar.gz && \
    cd binutils-2.33.1 && \
    ./configure && make -j && make install && cd .. && rm -rf binutils-2.33.1 binutils-2.33.1.tar.gz


# Install Go and glide
RUN wget -qO- https://paddle-ci.cdn.bcebos.com/go1.8.1.linux-amd64.tar.gz | \
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

RUN pip3 --no-cache-dir install pre-commit==1.10.4 ipython==5.3.0 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com && \
    pip3 --no-cache-dir install ipykernel==4.6.0 wheel -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# For PaddleTest CE
RUN pip3 --no-cache-dir install pytest -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

COPY requirements.txt /root/
RUN pip3 --no-cache-dir install -r /root/requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com


# Older versions of patchelf limited the size of the files being processed and were fixed in this pr.
# https://github.com/NixOS/patchelf/commit/ba2695a8110abbc8cc6baf0eea819922ee5007fa
# So install a newer version here.
RUN apt-get install software-properties-common && \
    apt-get update && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update -y && \
    apt install gcc-10 -y && \
    wget -q http://mirrors.edge.kernel.org/ubuntu/pool/universe/p/patchelf/patchelf_0.10-2build1_amd64.deb && \
    dpkg -i patchelf_0.10-2build1_amd64.deb

# Configure OpenSSH server. c.f. https://docs.docker.com/engine/examples/running_ssh_service
#RUN mkdir /var/run/sshd && echo 'root:root' | chpasswd && sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config && sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
#CMD source ~/.bashrc

# ccache 3.7.9
RUN wget https://paddle-ci.gz.bcebos.com/ccache-3.7.9.tar.gz && \
    tar xf ccache-3.7.9.tar.gz && mkdir /usr/local/ccache-3.7.9 && cd ccache-3.7.9 && \
    ./configure -prefix=/usr/local/ccache-3.7.9 && \
    make -j8 && make install && \
    ln -s /usr/local/ccache-3.7.9/bin/ccache /usr/local/bin/ccache

# For CINN environment
RUN apt update --fix-missing && \
    apt install autoconf autogen libtool zlib1g-dev sudo libginac-dev clang cmake -y && \
    apt remove python3-six python-six -y && \
    pip3 install numpy pybind11 six matplotlib && \
    update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2 && \
    python3 -m pip install paddlepaddle-gpu==2.1.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Install LLVM
RUN echo deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic main >> /etc/apt/source.list && \
    echo deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic main >> /etc/apt/source.list && \
    echo deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-10 main >> /etc/apt/source.list && \
    echo deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-10 main >> /etc/apt/source.list

RUN ln -s /usr/bin/llvm-config-6.0 /usr/bin/llvm-config && \
    printf "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-10 main" |tee /etc/apt/sources.list.d/llvm-toolchain-xenial-10.list && \
    wget -q -O - http://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add - && \
    apt install -y libclang-dev llvm-10 llvm-10-dev libclang-10-dev -y

# set C++ Path, libcudnn.so and llvm11 with mlir
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/include/c++/7:/usr/include/x86_64-linux-gnu/c++/7
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/libcudnn.so && \
    mkdir /WorkSpace && \
    cd /WorkSpace && \
    wget -q https://paddle-inference-dist.bj.bcebos.com/CINN/llvm11-latest.tar && \
    tar -xvf llvm11-latest.tar
ENV LLVM11_DIR=/WorkSpace/llvm11-latest

WORKDIR /WorkSpace
EXPOSE 22
