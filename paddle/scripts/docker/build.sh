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
if [[ ${BUILD_AND_INSTALL:-OFF} == 'ON' ]]; then
    if [[ ${WITH_GPU:-OFF} == 'ON' ]]; then
	ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/lib/libcudnn.so
    fi

    mkdir -p /paddle/build # -p means no error if exists
    # clean local cmake and third_party cache
    cd /paddle/build && rm -rf * && rm -rf ../third_party
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

trap : 0
