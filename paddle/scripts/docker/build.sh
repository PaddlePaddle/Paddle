#!/bin/bash

function abort(){
    echo "An error occurred. Exiting..." 1>&2
    exit 1
}

trap 'abort' 0
set -e

# If Dockerfile.* sets BUILD_AND_INSTALL to 'ON', it would have copied
# source tree to /paddle, and this scripts should build it into
# /paddle/build.
if [[ ${BUILD_AND_INSTALL:-ON} == 'ON' ]]; then
    if [[ ${WITH_GPU:-OFF} == 'ON' ]]; then
	ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/lib/libcudnn.so
    fi

    mkdir -p /paddle/build # -p means no error if exists
    cd /paddle/build
    cmake .. \
	  -DWITH_DOC=ON \
	  -DWITH_GPU=${WITH_GPU:-OFF} \
	  -DWITH_AVX=${WITH_AVX:-OFF} \
	  -DWITH_SWIG_PY=ON \
	  -DCUDNN_ROOT=/usr/ \
	  -DWITH_STYLE_CHECK=OFF \
	  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    make -j `nproc`
    make install

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

    pip install /usr/local/opt/paddle/share/wheels/*.whl
    paddle version
fi

trap : 0
