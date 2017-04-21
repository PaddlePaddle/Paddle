#!/bin/bash

set -e

# Set BASE_IMAGE according to env variables
if [ ${WITH_GPU} == "ON" ]; then
  BASE_IMAGE="nvidia/cuda:8.0-cudnn5-runtime-ubuntu14.04"
  # additional packages to install when building gpu images
  GPU_DOCKER_PKG="python-pip python-dev"
else
  BASE_IMAGE="python:2.7.13-slim"
  # FIXME: python base image uses different python version than WITH_GPU
  # need to change PYTHONHOME to /usr/local when using python base image
  CPU_DOCKER_PYTHON_HOME_ENV="ENV PYTHONHOME /usr/local"
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

# Since python v2 api import py_paddle module, the generation of paddle docs
# depend on paddle's compilation and installation
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
# FIXME(typhoonzero): should we remove paddle/scripts/deb ?
# FIXME: CPACK_DEBIAN_PACKAGE_DEPENDS removes all dev dependencies, must
# install them in docker
cpack -D CPACK_GENERATOR='DEB' -D CPACK_DEBIAN_PACKAGE_DEPENDS="" ..

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

if [[ -n ${APT_MIRROR} ]]; then
  MIRROR_UPDATE="sed -i '${APT_MIRROR}' /etc/apt/sources.list && \\"
else
  MIRROR_UPDATE="\\"
fi

cat > /paddle/build/Dockerfile <<EOF
FROM ${BASE_IMAGE}
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>
ENV HOME /root
ENV LANG en_US.UTF-8
# Use Fix locales to en_US.UTF-8
RUN ${MIRROR_UPDATE}
    apt-get update && \
    apt-get install -y libgfortran3 libpython2.7 ${GPU_DOCKER_PKG} && \
    apt-get clean -y && \
    pip install --upgrade pip && \
    pip install -U 'protobuf==3.1.0' requests numpy
# Use different deb file when building different type of images
ADD build/*.deb /usr/local/opt/paddle/deb/
# run paddle version to install python packages first
RUN dpkg -i /usr/local/opt/paddle/deb/*.deb && \
    rm -f /usr/local/opt/paddle/deb/*.deb && \
    pip install /usr/local/opt/paddle/share/wheels/*.whl && \
    paddle version
${CPU_DOCKER_PYTHON_HOME_ENV}
${DOCKERFILE_CUDNN_DSO}
${DOCKERFILE_GPU_ENV}
# default command shows the paddle version and exit
CMD ["paddle", "version"]
EOF
