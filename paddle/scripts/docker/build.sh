#!/bin/bash

set -xe

# Set BASE_IMAGE according to env variables
if [ ${WITH_GPU} == "ON" ]; then
  BASE_IMAGE="nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04"
else
  BASE_IMAGE="ubuntu:16.04"
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

cat <<EOF
========================================
Configuring cmake in /paddle/build ...
      -DCMAKE_BUILD_TYPE=Release
      -DWITH_DOC=OFF
      -DWITH_GPU=${WITH_GPU:-OFF}
      -DWITH_AVX=${WITH_AVX:-OFF}
      -DWITH_SWIG_PY=ON
      -DCUDNN_ROOT=/usr/
      -DWITH_STYLE_CHECK=${WITH_STYLE_CHECK:-OFF}
      -DWITH_TESTING=${WITH_TESTING:-OFF}
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
========================================
EOF
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

cat <<EOF
========================================
Building in /paddle/build ...
   Build unit tests: ${WITH_TESTING:-OFF}
========================================
EOF
make -j `nproc`
if [ ${WITH_TESTING:-OFF} == "ON" ] && [ ${RUN_TEST:-OFF} == "ON" ] ; then
    pip uninstall -y py-paddle paddle || true
    ctest -V
fi


cat <<EOF
========================================
Installing ...
========================================
EOF
make install
pip install /usr/local/opt/paddle/share/wheels/*.whl
paddle version


# To build documentation, we need to run cmake again after installing
# PaddlePaddle.  This awkwardness is due to
# https://github.com/PaddlePaddle/Paddle/issues/1854.  It also
# describes a solution.
if [ ${WITH_DOC} == "ON" ]; then
    cat <<EOF
========================================
Building documentation ...
   In /paddle/build_doc
========================================
EOF
    mkdir -p /paddle/build_doc
    pushd /paddle/build_doc
    cmake .. \
          -DWITH_DOC=ON \
          -DWITH_GPU=OFF \
          -DWITH_AVX=${WITH_AVX:-ON} \
          -DWITH_SWIG_PY=ON \
          -DWITH_STYLE_CHECK=OFF
    make paddle_docs paddle_docs_cn
    popd
fi


if [[ ${WOBOQ:-OFF} == 'ON' ]]; then
    cat <<EOF
========================================
Converting C++ source code into HTML ...
========================================
EOF
    export WOBOQ_OUT=/paddle/build/woboq_out
    mkdir -p $WOBOQ_OUT
    cp -rv /woboq/data $WOBOQ_OUT/../data
    /woboq/generator/codebrowser_generator \
        -b /paddle/build \
        -a \
        -o $WOBOQ_OUT \
        -p paddle:/paddle
    /woboq/indexgenerator/codebrowser_indexgenerator $WOBOQ_OUT
fi

# generate deb package for current build
# FIXME(typhoonzero): should we remove paddle/scripts/deb ?
cat <<EOF
========================================
Generating .deb package ...
========================================
EOF
cpack -D CPACK_GENERATOR='DEB' ..


cat <<EOF
========================================
Generate /paddle/build/Dockerfile ...
========================================
EOF

cat > /paddle/build/Dockerfile <<EOF
FROM ${BASE_IMAGE}
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>
ENV HOME /root
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
    apt-get install -y python-pip locales && \
    pip install -U pip && \
    dpkg -i /*.deb ; apt-get install -f -y && \
    apt-get clean -y && \
    rm -f /*.deb

# Fix locales to en_US.UTF-8
RUN localedef -i en_US -f UTF-8 en_US.UTF-8

${DOCKERFILE_CUDNN_DSO}
${DOCKERFILE_GPU_ENV}

# default command shows the paddle version and exit
CMD ["paddle", "version"]
EOF
