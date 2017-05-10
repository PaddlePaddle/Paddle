#!/bin/bash

set -e

# Set BASE_IMAGE according to env variables
if [ ${WITH_GPU} == "ON" ]; then
  BASE_IMAGE="nvidia/cuda:8.0-cudnn5-runtime-ubuntu14.04"
else
  BASE_IMAGE="ubuntu:14.04"
fi

DOCKERFILE_GPU_ENV=""
DOCKERFILE_CUDNN_DSO=""
if [[ ${WITH_GPU:-OFF} == 'ON' ]]; then
    DOCKERFILE_GPU_ENV="ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
    DOCKERFILE_CUDNN_DSO="RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.5 /usr/lib/x86_64-linux-gnu/libcudnn.so"
fi

mkdir -p /paddle/build
cd /paddle/build

# build script will not fail if *.deb does not exist
rm *.deb 2>/dev/null || true

cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_DOC=OFF \
      -DWITH_GPU=${WITH_GPU:-OFF} \
      -DWITH_AVX=${WITH_AVX:-OFF} \
      -DWITH_SWIG_PY=ON \
      -DCUDNN_ROOT=/usr/ \
      -DWITH_STYLE_CHECK=${WITH_STYLE_CHECK:-OFF} \
      -DWITH_TESTING=${WITH_TESTING:-OFF} \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j `nproc`
if [ ${WITH_TESTING:-OFF} == "ON" ] && [ ${RUN_TEST:-OFF} == "ON" ] ; then
    make test
fi
make install
pip install /usr/local/opt/paddle/share/wheels/*.whl

# To build documentation, we need to run cmake twice.
# This awkwardness is due to https://github.com/PaddlePaddle/Paddle/issues/1854.
# It also describes a solution.
if [ ${WITH_DOC} == "ON" ]; then
    mkdir -p /paddle/build_doc
    pushd /paddle/build_doc
    cmake .. \
          -DWITH_DOC=ON \
          -DWITH_GPU=OFF \
          -DWITH_AVX=${WITH_AVX:-OFF} \
          -DWITH_SWIG_PY=ON \
          -DWITH_STYLE_CHECK=OFF
    make paddle_docs paddle_docs_cn
    popd
fi
# generate deb package for current build
cpack -D CPACK_GENERATOR='DEB' ..

if [[ ${WOBOQ:-OFF} == 'ON' ]]; then
    apt-get install -y clang-3.8 llvm-3.8 libclang-3.8-dev
    # Install woboq_codebrowser.
    git clone https://github.com/woboq/woboq_codebrowser /woboq
    cd /woboq
    cmake -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-3.8 \
          -DCMAKE_BUILD_TYPE=Release \
          .
    make

    export WOBOQ_OUT=/woboq_out/paddle
    export BUILD_DIR=/paddle/build
    mkdir -p $WOBOQ_OUT
    cp -rv /woboq/data $WOBOQ_OUT/../data
    /woboq/generator/codebrowser_generator \
        -b /paddle/build \
        -a \
        -o $WOBOQ_OUT \
        -p paddle:/paddle
    /woboq/indexgenerator/codebrowser_indexgenerator $WOBOQ_OUT
    cd /woboq
    make clean
fi

paddle version

cat > /paddle/build/Dockerfile <<EOF
FROM ${BASE_IMAGE}
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>
ENV HOME /root
ENV LANG en_US.UTF-8
# Use Fix locales to en_US.UTF-8
EOF

if [[ -n ${APT_MIRROR} ]]; then
cat >> /paddle/build/Dockerfile <<EOF
RUN sed -i '${APT_MIRROR}' /etc/apt/sources.list
EOF
fi

cat >> /paddle/build/Dockerfile <<EOF
# Use different deb file when building different type of images
ADD *.deb /
# run paddle version to install python packages first
RUN apt-get update &&\
    apt-get install -y python-pip && pip install -U pip && \
    dpkg -i /*.deb ; apt-get install -f -y && \
    apt-get clean -y && \
    rm -f /*.deb && \
    paddle version
${DOCKERFILE_CUDNN_DSO}
${DOCKERFILE_GPU_ENV}
# default command shows the paddle version and exit
CMD ["paddle", "version"]
EOF
