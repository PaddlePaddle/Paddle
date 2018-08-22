FROM ubuntu:16.04


# anakin install basic tools
RUN apt-get update
RUN apt-get install -y \
            build-essential \
            vim \
            wget \
            git \
            make \
            gcc \
            libiomp-dev \
            libopenmpi-dev \
            openmpi-bin \
            openmpi-doc \
            python-dev \
            python-pip \
            && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --upgrade pip==9.0.3 && \
    pip install --no-cache-dir \
        flask numpy pyyaml scipy pandas



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
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# clone anakin source and build on x86
RUN git clone --branch developing --recursive "https://github.com/PaddlePaddle/Anakin.git" && cd Anakin/tools/ && ./x86_build.sh && cd -

