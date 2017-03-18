#!/bin/bash

function abort(){
    echo "An error occurred. Exiting..." 1>&2
    exit 1
}

trap 'abort' 0
set -e
mkdir -p /paddle/dist/cpu
mkdir -p /paddle/dist/gpu
mkdir -p /paddle/dist/cpu-noavx
mkdir -p /paddle/dist/gpu-noavx
# Set BASE_IMAGE and DEB_PATH according to env variables
if [ ${WITH_GPU} == "ON" ]; then
  BASE_IMAGE="nvidia/cuda:7.5-cudnn5-runtime-ubuntu14.04"
  # additional packages to install when building gpu images
  GPU_DOCKER_PKG="python-pip"
  if [ ${WITH_AVX} == "ON" ]; then
    DEB_PATH="dist/gpu/"
    DOCKER_SUFFIX="gpu"
  else
    DEB_PATH="dist/gpu-noavx/"
    DOCKER_SUFFIX="gpu-noavx"
  fi
else
  BASE_IMAGE="python:2.7.13-slim"
  if [ ${WITH_AVX} == "ON" ]; then
    DEB_PATH="dist/cpu/"
    DOCKER_SUFFIX="cpu"
  else
    DEB_PATH="dist/cpu-noavx/"
    DOCKER_SUFFIX="noavx"
  fi
fi
# If Dockerfile.* sets BUILD_AND_INSTALL to 'ON', it would have copied
# source tree to /paddle, and this scripts should build it into
# /paddle/build.
if [[ ${BUILD_AND_INSTALL:-OFF} == 'ON' ]]; then
    if [[ ${WITH_GPU:-OFF} == 'ON' ]]; then
	ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/lib/libcudnn.so
    fi

    mkdir -p /paddle/build # -p means no error if exists
    cd /paddle/build
    # clean local cmake and third_party cache
    if [ ${DELETE_BUILD_CACHE} == 'ON' ]; then
      rm -rf * && rm -rf ../third_party
    fi
    cmake .. \
	  -DWITH_DOC=${WITH_DOC:-OFF} \
	  -DWITH_GPU=${WITH_GPU:-OFF} \
	  -DWITH_AVX=${WITH_AVX:-OFF} \
	  -DWITH_SWIG_PY=ON \
	  -DCUDNN_ROOT=/usr/ \
	  -DWITH_STYLE_CHECK=OFF \
	  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    make -j `nproc`
    make install
    # generate deb package for current build
    # FIXME(typhoonzero): should we remove paddle/scripts/deb ?
    # FIXME: CPACK_DEBIAN_PACKAGE_DEPENDS removes all dev dependencies, must
    # install them in docker
    cpack -D CPACK_GENERATOR='DEB' -D CPACK_DEBIAN_PACKAGE_DEPENDS="" ..
    mv /paddle/build/*.deb /paddle/${DEB_PATH}

    if [[ ${BUILD_WOBOQ:-OFF} == 'ON' ]]; then
        apt-get install -y clang-3.8 llvm-3.8 libclang-3.8-dev
        # Install woboq_codebrowser.
        git clone https://github.com/woboq/woboq_codebrowser /woboq
        cd /woboq
        cmake -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-3.8 \
        -DCMAKE_BUILD_TYPE=Release \
        .
        make

        export WOBOQ_OUT=/usr/share/nginx/html/paddle
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

    pip install /usr/local/opt/paddle/share/wheels/py_paddle*linux*.whl
    pip install /usr/local/opt/paddle/share/wheels/paddle*.whl
    paddle version

    if [[ ${DOCKER_BUILD:-FALSE} == 'TRUE' ]]; then
	# reduce docker image size
	rm -rf /paddle/build
	rm -rf /usr/local/opt/paddle/share/wheels/
    fi
fi

# generate production docker image Dockerfile
if [ ${USE_MIRROR} ]; then
  MIRROR_UPDATE="sed 's@http:\/\/archive.ubuntu.com\/ubuntu\/@mirror:\/\/mirrors.ubuntu.com\/mirrors.txt@' -i /etc/apt/sources.list && \\"
else
  MIRROR_UPDATE="\\"
fi

cat > /paddle/build/Dockerfile.${DOCKER_SUFFIX} <<EOF
FROM ${BASE_IMAGE}
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>

# ENV variables
ARG WITH_AVX
ARG WITH_DOC
ARG WITH_STYLE_CHECK

ENV WITH_GPU=${WITH_GPU}
ENV WITH_AVX=\${WITH_AVX:-ON}
ENV WITH_DOC=\${WITH_DOC:-OFF}
ENV WITH_STYLE_CHECK=\${WITH_STYLE_CHECK:-OFF}

ENV HOME /root
ENV LANG en_US.UTF-8

# Use Fix locales to en_US.UTF-8

RUN ${MIRROR_UPDATE}
    apt-get update && \
    apt-get install -y libgfortran3 ${GPU_DOCKER_PKG} && \
    apt-get clean -y && \
    pip install --upgrade pip && \
    pip install -U 'protobuf==3.1.0' requests
RUN pip install numpy
# Use different deb file when building different type of images
ADD \$PWD/${DEB_PATH}*.deb /usr/local/opt/paddle/deb/
RUN dpkg --force-all -i /usr/local/opt/paddle/deb/*.deb && rm -f /usr/local/opt/paddle/deb/*.deb

ENV PATH="/usr/local/opt/paddle/bin/:${PATH}"
# default command shows the paddle version and exit
CMD ["paddle", "version"]
EOF

trap : 0
