FROM nvidia/cuda:8.0-cudnn7-devel-centos7

# anakin install ubuntu GPU env
RUN yum-config-manager --disable cuda && yum -y install vim wget git make glibc-devel libstdc++-deve epel-release && rm -rf /var/cache/yum/*

RUN yum -y install python-pip && rm -rf /var/cache/yum/*

RUN pip install  --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        flask numpy pyyaml scipy pandas

# set env
ENV LIBRARY_PATH /usr/lib64:$LIBRARY_PATH

# install cmake
RUN wget https://cmake.org/files/v3.2/cmake-3.2.0.tar.gz && tar xzf cmake-3.2.0.tar.gz && \
    cd cmake-3.2.0 && ./bootstrap && \
    make -j4 && make install && cd .. && rm -f cmake-3.2.0.tar.gz

# install protobuf
RUN wget --no-check-certificate https://mirror.sobukus.de/files/src/protobuf/protobuf-cpp-3.4.0.tar.gz \
    && tar -xvf protobuf-cpp-3.4.0.tar.gz \
    && cd protobuf-3.4.0 && ./configure \
    && make -j4 && make install && cd .. \
    && rm -f protobuf-cpp-3.4.0.tar.gz

# set env
ENV CUDNN_ROOT /usr/local/cuda/include

# build and install anakin
RUN git clone --branch developing --recursive https://github.com/PaddlePaddle/Anakin.git
