FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

# anakin install ubuntu GPU env
RUN apt-get update ; apt-get install -y\
        build-essential \
        cmake \
        git \
        libiomp-dev \
        libopencv-dev \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-doc \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy \
        wget \
        &&  rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip==9.0.3 && \
    pip install --no-cache-dir \
        flask \
        numpy \
        pyyaml \
        scipy \
        pandas

# install cmake
RUN wget https://cmake.org/files/v3.2/cmake-3.2.0.tar.gz && tar xzf cmake-3.2.0.tar.gz && \
        cd cmake-3.2.0 && ./bootstrap && \
        make -j4 && make install && cd .. && rm -f cmake-3.2.0.tar.gz


# install protobuf
RUN wget --no-check-certificate https://mirror.sobukus.de/files/src/protobuf/protobuf-cpp-3.4.0.tar.gz \
                        && tar -xvf protobuf-cpp-3.4.0.tar.gz \
                        && cd protobuf-3.4.0 && ./configure \
                        && make -j4 && make install && cd .. && rm -f protobuf-cpp-3.4.0.tar.gz

# set env
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

ENV CUDNN_ROOT=/usr/local/cuda/include

# build and install anakin
RUN git clone --branch developing --recursive https://github.com/PaddlePaddle/Anakin.git 
