#!/bin/bash

function cmake_gen() {
    mkdir -p /paddle/build
    cd /paddle/build

    # build script will not fail if *.deb does not exist
    rm *.deb 2>/dev/null || true
    # delete previous built whl packages
    rm -rf /paddle/paddle/dist 2>/dev/null || true

    # Support build for all python versions, currently
    # including cp27-cp27m and cp27-cp27mu.
    PYTHON_FLAGS=""
    if [ "$1" != "" ]; then
        echo "using python abi: $1"
        if [ "$1" == "cp27-cp27m" ]; then
            export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs2/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs4/lib:}
            export PATH=/opt/python/cp27-cp27m/bin/:${PATH}
            PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27m/bin/python
        -DPYTHON_INCLUDE_DIR:PATH=/opt/python/cp27-cp27m/include/python2.7
        -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-2.7.11-ucs2/lib/libpython2.7.so"
        elif [ "$1" == "cp27-cp27mu" ]; then
            export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs2/lib:}
            export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
            PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27mu/bin/python
        -DPYTHON_INCLUDE_DIR:PATH=/opt/python/cp27-cp27mu/include/python2.7
        -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-2.7.11-ucs4/lib/libpython2.7.so"
        fi
    fi

    cat <<EOF
    ========================================
    Configuring cmake in /paddle/build ...
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
        ${PYTHON_FLAGS}
        -DWITH_DSO=ON
        -DWITH_DOC=${WITH_DOC:-OFF}
        -DWITH_GPU=${WITH_GPU:-OFF}
        -DWITH_AMD_GPU=${WITH_AMD_GPU:-OFF}
        -DWITH_DISTRIBUTE=${WITH_DISTRIBUTE:-OFF}
        -DWITH_MKL=${WITH_MKL:-ON}
        -DWITH_AVX=${WITH_AVX:-OFF}
        -DWITH_GOLANG=${WITH_GOLANG:-OFF}
        -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All}
        -DWITH_SWIG_PY=ON
        -DWITH_C_API=${WITH_C_API:-OFF}
        -DWITH_PYTHON=${WITH_PYTHON:-ON}
        -DWITH_SWIG_PY=${WITH_SWIG_PY:-ON}
        -DCUDNN_ROOT=/usr/
        -DWITH_STYLE_CHECK=${WITH_STYLE_CHECK:-ON}
        -DWITH_TESTING=${WITH_TESTING:-ON}
        -DWITH_FAST_BUNDLE_TEST=ON
        -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DWITH_FLUID_ONLY=${WITH_FLUID_ONLY:-OFF}
    ========================================
EOF
    # Disable UNITTEST_USE_VIRTUALENV in docker because
    # docker environment is fully controlled by this script.
    # See /Paddle/CMakeLists.txt, UNITTEST_USE_VIRTUALENV option.
    cmake .. \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} \
        ${PYTHON_FLAGS} \
        -DWITH_DSO=ON \
        -DWITH_DOC=${WITH_DOC:-OFF} \
        -DWITH_GPU=${WITH_GPU:-OFF} \
        -DWITH_AMD_GPU=${WITH_AMD_GPU:-OFF} \
        -DWITH_DISTRIBUTE=${WITH_DISTRIBUTE:-OFF} \
        -DWITH_MKL=${WITH_MKL:-ON} \
        -DWITH_AVX=${WITH_AVX:-OFF} \
        -DWITH_GOLANG=${WITH_GOLANG:-OFF} \
        -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All} \
        -DWITH_SWIG_PY=${WITH_SWIG_PY:-ON} \
        -DWITH_C_API=${WITH_C_API:-OFF} \
        -DWITH_PYTHON=${WITH_PYTHON:-ON} \
        -DCUDNN_ROOT=/usr/ \
        -DWITH_STYLE_CHECK=${WITH_STYLE_CHECK:-ON} \
        -DWITH_TESTING=${WITH_TESTING:-ON} \
        -DWITH_FAST_BUNDLE_TEST=ON \
        -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake \
        -DWITH_FLUID_ONLY=${WITH_FLUID_ONLY:-OFF} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
}

function run_build() {
    cat <<EOF
    ============================================
    Building in /paddle/build ...
    ============================================
EOF
    make clean
    make -j `nproc`
}

function run_test() {
    if [ ${WITH_TESTING:-ON} == "ON" ] && [ ${RUN_TEST:-OFF} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests ...
    ========================================
EOF
        ctest --output-on-failure
        # make install should also be test when unittest
        make install -j `nproc`
        pip install /usr/local/opt/paddle/share/wheels/*.whl
        if [[ ${WITH_FLUID_ONLY:-OFF} == "OFF" ]] ; then
            paddle version
        fi
    fi
}


function gen_docs() {
    if [[ ${WITH_DOC:-OFF} == "ON" ]]; then
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

        make -j `nproc` paddle_docs paddle_apis
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
}


function gen_dockerfile() {
    # Set BASE_IMAGE according to env variables
    if [[ ${WITH_GPU} == "ON" ]]; then
    BASE_IMAGE="nvidia/cuda:8.0-cudnn7-runtime-ubuntu16.04"
    else
    BASE_IMAGE="ubuntu:16.04"
    fi

    DOCKERFILE_GPU_ENV=""
    DOCKERFILE_CUDNN_DSO=""
    if [[ ${WITH_GPU:-OFF} == 'ON' ]]; then
        DOCKERFILE_GPU_ENV="ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH}"
        DOCKERFILE_CUDNN_DSO="RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/lib/x86_64-linux-gnu/libcudnn.so"
    fi

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

    if [[ ${WITH_GPU} == "ON"  ]]; then
        NCCL_DEPS="apt-get install -y libnccl2=2.1.2-1+cuda8.0 libnccl-dev=2.1.2-1+cuda8.0 &&"
    else
        NCCL_DEPS=""
    fi

    if [[ ${WITH_FLUID_ONLY:-OFF} == "OFF" ]]; then
        PADDLE_VERSION="paddle version"
        CMD='"paddle", "version"'
    else
        PADDLE_VERSION="true"
        CMD='"true"'
    fi

    cat >> /paddle/build/Dockerfile <<EOF
    ADD python/dist/*.whl /
    # run paddle version to install python packages first
    RUN apt-get update &&\
        ${NCCL_DEPS}\
        apt-get install -y wget python-pip dmidecode python-tk && pip install -U pip==9.0.3 && \
        pip install /*.whl; apt-get install -f -y && \
        apt-get clean -y && \
        rm -f /*.whl && \
        ${PADDLE_VERSION} && \
        ldconfig
    ${DOCKERFILE_CUDNN_DSO}
    ${DOCKERFILE_GPU_ENV}
    ENV NCCL_LAUNCH_MODE PARALLEL
EOF
    if [[ ${WITH_GOLANG:-OFF} == "ON" ]]; then
        cat >> /paddle/build/Dockerfile <<EOF
        ADD go/cmd/pserver/pserver /usr/bin/
        ADD go/cmd/master/master /usr/bin/
EOF
    fi
    cat >> /paddle/build/Dockerfile <<EOF
    # default command shows the paddle version and exit
    CMD [${CMD}]
EOF
}

function gen_capi_package() {
  if [[ ${WITH_C_API} == "ON" ]]; then
    install_prefix="/paddle/build/capi_output"
    rm -rf $install_prefix
    make DESTDIR="$install_prefix" install
    cd $install_prefix/usr/local
    ls | egrep -v "^Found.*item$" | xargs tar -cf /paddle/build/paddle.tgz
  fi
}

function gen_fluid_inference_lib() {
    if [ ${WITH_C_API:-OFF} == "OFF" ] ; then
    cat <<EOF
    ========================================
    Deploying fluid inference library ...
    ========================================
EOF
        make -j `nproc` inference_lib_dist
    fi
}

set -xe

cmake_gen ${PYTHON_ABI:-""}
run_build
run_test
gen_docs
gen_dockerfile
gen_capi_package
gen_fluid_inference_lib

if [[ ${WITH_C_API:-OFF} == "ON" ]]; then
  printf "PaddlePaddle C-API libraries was generated on build/paddle.tgz\n"
else
  printf "If you need to install PaddlePaddle in develop docker image,"
  printf "please make install or pip install build/python/dist/*.whl.\n"
fi
