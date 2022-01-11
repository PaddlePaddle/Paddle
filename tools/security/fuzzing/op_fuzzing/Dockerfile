FROM ubuntu:18.04
CMD ["bash"]
LABEL maintainer="Yakun Zhang <zhangyakun02@baidu.com>"
RUN apt-get update && \
    apt-get install -y python3.8 python3.8-dev python3.8-distutils git wget patchelf software-properties-common && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main" && \
    apt-get install -y clang-12 && \
    wget -O - https://bootstrap.pypa.io/get-pip.py | python3.8
RUN CLANG_BIN="/usr/bin/clang-12" python3.8 -m pip install --no-cache-dir atheris && \
    python3.8 -m pip install numpy protobuf
RUN wget -q https://cmake.org/files/v3.16/cmake-3.16.0-Linux-x86_64.tar.gz && \
    tar -zxf cmake-3.16.0-Linux-x86_64.tar.gz && \
    rm cmake-3.16.0-Linux-x86_64.tar.gz

ENV PATH=/cmake-3.16.0-Linux-x86_64/bin:$PATH
ENV PYTHON_EXECUTABLE=/usr/bin/python3.8
ENV PYTHON_INCLUDE_DIRS=/usr/include/python3.8
ENV PYTHON_LIBRARY=/usr/lib/python3.8
ENV CC=/usr/bin/clang-12
ENV CXX=/usr/bin/clang++-12

COPY Paddle/Patches /Patches/
RUN git clone https://github.com/PaddlePaddle/Paddle.git && \
    cd Paddle && \
    git checkout develop && \
    git pull && \
    git checkout 249081b6ee9ada225c2aa3779a6935be65bc04e0 && \
    patch -p0 cmake/flags.cmake < /Patches/flags.cmake.patch && \
    patch -p0 cmake/external/pybind11.cmake < /Patches/pybind11.cmake.patch && \
    patch -p0 paddle/fluid/platform/init.cc < /Patches/init.cc.patch

RUN cd Paddle && mkdir build && cd build && cmake .. -DPY_VERSION=3.8 -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}  \
    -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DSANITIZER_TYPE=Address  \
    -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_AVX=OFF -DWITH_MKL=OFF && \
    make -j$(nproc) && python3.8 -m pip install -U python/dist/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl