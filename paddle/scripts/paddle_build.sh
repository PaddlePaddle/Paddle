#!/usr/bin/env bash

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#=================================================
#                   Utils
#=================================================

set -ex

function print_usage() {
    echo -e "\n${RED}Usage${NONE}:
    ${BOLD}${SCRIPT_NAME}${NONE} [OPTION]"

    echo -e "\n${RED}Options${NONE}:
    ${BLUE}build${NONE}: run build for x86 platform
    ${BLUE}build_android${NONE}: run build for android platform
    ${BLUE}build_ios${NONE}: run build for ios platform
    ${BLUE}test${NONE}: run all unit tests
    ${BLUE}single_test${NONE}: run a single unit test
    ${BLUE}bind_test${NONE}: parallel tests bind to different GPU
    ${BLUE}doc${NONE}: generate paddle documents
    ${BLUE}gen_doc_lib${NONE}: generate paddle documents library
    ${BLUE}html${NONE}: convert C++ source code into HTML
    ${BLUE}dockerfile${NONE}: generate paddle release dockerfile
    ${BLUE}capi${NONE}: generate paddle CAPI package
    ${BLUE}fluid_inference_lib${NONE}: deploy fluid inference library
    ${BLUE}check_style${NONE}: run code style check
    ${BLUE}cicheck${NONE}: run CI tasks
    ${BLUE}assert_api_not_changed${NONE}: check api compability
    "
}

function init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'

    PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
    if [ -z "${SCRIPT_NAME}" ]; then
        SCRIPT_NAME=$0
    fi
}

function cmake_gen() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build

    # build script will not fail if *.deb does not exist
    rm *.deb 2>/dev/null || true
    # delete previous built whl packages
    rm -rf python/dist 2>/dev/null || true

    # Support build for all python versions, currently
    # including cp27-cp27m and cp27-cp27mu.
    PYTHON_FLAGS=""
    SYSTEM=`uname -s`
    if [ "$SYSTEM" == "Darwin" ]; then
        echo "Using python abi: $1"
        if [[ "$1" == "cp27-cp27m" ]] || [[ "$1" == "" ]]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/2.7" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/2.7
                export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/2.7
                export PATH=/Library/Frameworks/Python.framework/Versions/2.7/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/2.7/bin/python2.7
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib"
            else
                exit 1
            fi
        elif [ "$1" == "cp35-cp35m" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.5" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.5/lib/
                export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.5/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.5/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.5/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.5/include/python3.5m/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.5/lib/libpython3.5m.dylib"
                WITH_FLUID_ONLY=${WITH_FLUID_ONLY:-ON}
            else
                exit 1
            fi
        fi
    else
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
            elif [ "$1" == "cp35-cp35m" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.5.1/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.5.1/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.5.1/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.5.1/include/python3.5m
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.5.1/lib/libpython3.so"
           fi
        fi
    fi

    if [ "$SYSTEM" == "Darwin" ]; then
        WITH_DISTRIBUTE=${WITH_DISTRIBUTE:-ON}
        WITH_AVX=${WITH_AVX:-ON}
        INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR:-~/.cache/inference_demo}
    else
        INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR:-/root/.cache/inference_demo}
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
        -DWITH_C_API=${WITH_C_API:-OFF}
        -DWITH_PYTHON=${WITH_PYTHON:-ON}
        -DWITH_SWIG_PY=${WITH_SWIG_PY:-ON}
        -DCUDNN_ROOT=/usr/
        -DWITH_TESTING=${WITH_TESTING:-ON}
        -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DWITH_FLUID_ONLY=${WITH_FLUID_ONLY:-OFF}
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DWITH_CONTRIB=${WITH_CONTRIB:-ON}
        -DWITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON}
        -DINFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR}
        -DWITH_ANAKIN=${WITH_ANAKIN:-OFF}
        -DPY_VERSION=${PY_VERSION:-2.7}
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build}
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
        -DWITH_TESTING=${WITH_TESTING:-ON} \
        -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake \
        -DWITH_FLUID_ONLY=${WITH_FLUID_ONLY:-OFF} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DWITH_CONTRIB=${WITH_CONTRIB:-ON} \
        -DWITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON} \
        -DINFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR} \
        -DWITH_ANAKIN=${WITH_ANAKIN:-OFF} \
        -DPY_VERSION=${PY_VERSION:-2.7} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build}

}

function abort(){
    echo "Your change doesn't follow PaddlePaddle's code style." 1>&2
    echo "Please use pre-commit to check what is wrong." 1>&2
    exit 1
}

function check_style() {
    trap 'abort' 0
    set -e

    if [ -x "$(command -v gimme)" ]; then
    	eval "$(GIMME_GO_VERSION=1.8.3 gimme)"
    fi

    # set up go environment for running gometalinter
    mkdir -p $GOPATH/src/github.com/PaddlePaddle/
    ln -sf ${PADDLE_ROOT} $GOPATH/src/github.com/PaddlePaddle/Paddle
    mkdir -p ./build/go
    cp go/glide.* build/go
    cd build/go; glide install; cd -

    export PATH=/usr/bin:$PATH
    pre-commit install
    clang-format --version

    if ! pre-commit run -a ; then
        git diff
        exit 1
    fi

    trap : 0
}

#=================================================
#              Build
#=================================================

function build() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    cat <<EOF
    ============================================
    Building in /paddle/build ...
    ============================================
EOF
    make clean
    make -j `nproc`
    make install -j `nproc`
}

function build_mac() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    cat <<EOF
    ============================================
    Building in /paddle/build ...
    ============================================
EOF
    make clean
    make -j 8
    make install -j 8
}

function build_android() {
    if [ $ANDROID_ABI == "arm64-v8a" ]; then
      ANDROID_ARCH=arm64
      if [ $ANDROID_API -lt 21 ]; then
        echo "Warning: arm64-v8a requires ANDROID_API >= 21."
        ANDROID_API=21
      fi
    else # armeabi, armeabi-v7a
      ANDROID_ARCH=arm
    fi

    ANDROID_STANDALONE_TOOLCHAIN=$ANDROID_TOOLCHAINS_DIR/$ANDROID_ARCH-android-$ANDROID_API

    cat <<EOF
    ============================================
    Generating the standalone toolchain ...
    ${ANDROID_NDK_HOME}/build/tools/make-standalone-toolchain.sh
          --arch=$ANDROID_ARCH
          --platform=android-$ANDROID_API
          --install-dir=${ANDROID_STANDALONE_TOOLCHAIN}
    ============================================
EOF
    ${ANDROID_NDK_HOME}/build/tools/make-standalone-toolchain.sh \
          --arch=$ANDROID_ARCH \
          --platform=android-$ANDROID_API \
          --install-dir=$ANDROID_STANDALONE_TOOLCHAIN

    BUILD_ROOT=${PADDLE_ROOT}/build_android
    DEST_ROOT=${PADDLE_ROOT}/install_android

    mkdir -p $BUILD_ROOT
    cd $BUILD_ROOT

    if [ $ANDROID_ABI == "armeabi-v7a" ]; then
      cmake -DCMAKE_SYSTEM_NAME=Android \
            -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_STANDALONE_TOOLCHAIN \
            -DANDROID_ABI=$ANDROID_ABI \
            -DANDROID_ARM_NEON=ON \
            -DANDROID_ARM_MODE=ON \
            -DHOST_C_COMPILER=/usr/bin/gcc \
            -DHOST_CXX_COMPILER=/usr/bin/g++ \
            -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
            -DCMAKE_BUILD_TYPE=MinSizeRel \
            -DUSE_EIGEN_FOR_BLAS=ON \
            -DWITH_C_API=ON \
            -DWITH_SWIG_PY=OFF \
            ..
    elif [ $ANDROID_ABI == "arm64-v8a" ]; then
      cmake -DCMAKE_SYSTEM_NAME=Android \
            -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_STANDALONE_TOOLCHAIN \
            -DANDROID_ABI=$ANDROID_ABI \
            -DANDROID_ARM_MODE=ON \
            -DHOST_C_COMPILER=/usr/bin/gcc \
            -DHOST_CXX_COMPILER=/usr/bin/g++ \
            -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
            -DCMAKE_BUILD_TYPE=MinSizeRel \
            -DUSE_EIGEN_FOR_BLAS=OFF \
            -DWITH_C_API=ON \
            -DWITH_SWIG_PY=OFF \
            ..
    elif [ $ANDROID_ABI == "armeabi" ]; then
      cmake -DCMAKE_SYSTEM_NAME=Android \
            -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_STANDALONE_TOOLCHAIN \
            -DANDROID_ABI=$ANDROID_ABI \
            -DANDROID_ARM_MODE=ON \
            -DHOST_C_COMPILER=/usr/bin/gcc \
            -DHOST_CXX_COMPILER=/usr/bin/g++ \
            -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
            -DCMAKE_BUILD_TYPE=MinSizeRel \
            -DWITH_C_API=ON \
            -DWITH_SWIG_PY=OFF \
            ..
    else
      echo "Invalid ANDROID_ABI: $ANDROID_ABI"
    fi

    cat <<EOF
    ============================================
    Building in $BUILD_ROOT ...
    ============================================
EOF
    make -j `nproc`
    make install -j `nproc`
}

function build_ios() {
    # Create the build directory for CMake.
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build

    # Compile paddle binaries
    cmake .. \
          -DCMAKE_SYSTEM_NAME=iOS \
          -DIOS_PLATFORM=OS \
          -DCMAKE_OSX_ARCHITECTURES="arm64" \
          -DWITH_C_API=ON \
          -DUSE_EIGEN_FOR_BLAS=ON \
          -DWITH_TESTING=OFF \
          -DWITH_SWIG_PY=OFF \
          -DCMAKE_BUILD_TYPE=Release

    make -j 2
}

function run_test() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests ...
    ========================================
EOF
        ctest --output-on-failure
        # make install should also be test when unittest
        make install -j `nproc`
        pip install ${INSTALL_PREFIX:-/paddle/build}/opt/paddle/share/wheels/*.whl
        if [[ ${WITH_FLUID_ONLY:-OFF} == "OFF" ]] ; then
            paddle version
        fi
    fi
}

function run_mac_test() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests ...
    ========================================
EOF
        #remove proxy here to fix dist error on mac
        export http_proxy=
        export https_proxy=
        # TODO: jiabin need to refine this part when these tests fixed on mac
        ctest --output-on-failure -j $1
        # make install should also be test when unittest
        make install -j 8
        pip install --user ${INSTALL_PREFIX:-/paddle/build}/opt/paddle/share/wheels/*.whl
        if [[ ${WITH_FLUID_ONLY:-OFF} == "OFF" ]] ; then
            paddle version
        fi
        pip uninstall -y paddlepaddle
    fi
}

function assert_api_not_changed() {
    mkdir -p ${PADDLE_ROOT}/build/.check_api_workspace
    cd ${PADDLE_ROOT}/build/.check_api_workspace
    virtualenv .env
    source .env/bin/activate
    pip install ${PADDLE_ROOT}/build/python/dist/*whl
    python ${PADDLE_ROOT}/tools/print_signatures.py paddle.fluid > new.spec
    if [ "$1" == "cp35-cp35m" ]; then
        # Use sed to make python2 and python3 sepc keeps the same
        sed -i 's/arg0: str/arg0: unicode/g' new.spec
        sed -i "s/\(.*Transpiler.*\).__init__ ArgSpec(args=\['self'].*/\1.__init__ /g" new.spec
    fi
    python ${PADDLE_ROOT}/tools/diff_api.py ${PADDLE_ROOT}/paddle/fluid/API.spec new.spec
    deactivate
}

function assert_api_spec_approvals() {
    if [ -z ${BRANCH} ]; then
        BRANCH="develop"
    fi

    API_CHANGE=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/API.spec" || true`
    echo "checking API.spec change, PR: ${GIT_PR_ID}, changes: ${API_CHANGE}"
    if [ ${API_CHANGE} ] && [ "${GIT_PR_ID}" != "" ]; then
        # NOTE: per_page=10000 should be ok for all cases, a PR review > 10000 is not human readable.
        APPROVALS=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000 | \
        python ${PADDLE_ROOT}/tools/check_pr_approval.py 2 7845005 2887803 728699 13348433`
        echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
        if [ "${APPROVALS}" == "FALSE" ]; then
            echo "You must have at least 2 approvals for the api change!"
        exit 1
        fi
    fi
}


function single_test() {
    TEST_NAME=$1
    if [ -z "${TEST_NAME}" ]; then
        echo -e "${RED}Usage:${NONE}"
        echo -e "${BOLD}${SCRIPT_NAME}${NONE} ${BLUE}single_test${NONE} [test_name]"
        exit 1
    fi
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running ${TEST_NAME} ...
    ========================================
EOF
        ctest --output-on-failure -R ${TEST_NAME}
    fi
}

function bind_test() {
    # the number of process to run tests
    NUM_PROC=6

    # calculate and set the memory usage for each process
    MEM_USAGE=$(printf "%.2f" `echo "scale=5; 1.0 / $NUM_PROC" | bc`)
    export FLAGS_fraction_of_gpu_memory_to_use=$MEM_USAGE

    # get the CUDA device count
    CUDA_DEVICE_COUNT=$(nvidia-smi -L | wc -l)

    for (( i = 0; i < $NUM_PROC; i++ )); do
        cuda_list=()
        for (( j = 0; j < $CUDA_DEVICE_COUNT; j++ )); do
            s=$[i+j]
            n=$[s%CUDA_DEVICE_COUNT]
            if [ $j -eq 0 ]; then
                cuda_list=("$n")
            else
                cuda_list="$cuda_list,$n"
            fi
        done
        echo $cuda_list
        # CUDA_VISIBLE_DEVICES http://acceleware.com/blog/cudavisibledevices-masking-gpus
        # ctest -I https://cmake.org/cmake/help/v3.0/manual/ctest.1.html?highlight=ctest
        env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC --output-on-failure &
    done
    wait
}


function gen_docs() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    cat <<EOF
    ========================================
    Building documentation ...
    In /paddle/build
    ========================================
EOF
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_DOC=ON \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF

    make -j `nproc` paddle_docs paddle_apis

    # check websites for broken links
    linkchecker doc/v2/en/html/index.html
    linkchecker doc/v2/cn/html/index.html
    linkchecker doc/v2/api/en/html/index.html

}

function gen_doc_lib() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    cat <<EOF
    ========================================
    Building documentation library ...
    In /paddle/build
    ========================================
EOF
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_DOC=ON \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DWITH_FLUID_ONLY=ON

    local LIB_TYPE=$1
    case $LIB_TYPE in
      full)
        # Build full Paddle Python module. Will timeout without caching 'copy_paddle_pybind' first
        make -j `nproc` gen_proto_py framework_py_proto copy_paddle_pybind paddle_python
        ;;
      pybind)
        # Build paddle pybind library. Takes 49 minutes to build. Might timeout
        make -j `nproc` copy_paddle_pybind
        ;;
      proto)
        # Even smaller library.
        make -j `nproc` framework_py_proto
        ;;
      *)
        exit 0
        ;;
      esac
}

function gen_html() {
    cat <<EOF
    ========================================
    Converting C++ source code into HTML ...
    ========================================
EOF
    export WOBOQ_OUT=${PADDLE_ROOT}/build/woboq_out
    mkdir -p $WOBOQ_OUT
    cp -rv /woboq/data $WOBOQ_OUT/../data
    /woboq/generator/codebrowser_generator \
    	-b ${PADDLE_ROOT}/build \
    	-a \
    	-o $WOBOQ_OUT \
    	-p paddle:${PADDLE_ROOT}
    /woboq/indexgenerator/codebrowser_indexgenerator $WOBOQ_OUT
}

function gen_dockerfile() {
    # Set BASE_IMAGE according to env variables
    CUDA_MAJOR="$(echo $CUDA_VERSION | cut -d '.' -f 1).$(echo $CUDA_VERSION | cut -d '.' -f 2)"
    CUDNN_MAJOR=$(echo $CUDNN_VERSION | cut -d '.' -f 1)
    if [[ ${WITH_GPU} == "ON" ]]; then
        BASE_IMAGE="nvidia/cuda:${CUDA_MAJOR}-cudnn${CUDNN_MAJOR}-runtime-ubuntu16.04"
    else
        BASE_IMAGE="ubuntu:16.04"
    fi

    DOCKERFILE_GPU_ENV=""
    DOCKERFILE_CUDNN_DSO=""
    DOCKERFILE_CUBLAS_DSO=""
    if [[ ${WITH_GPU:-OFF} == 'ON' ]]; then
        DOCKERFILE_GPU_ENV="ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH}"
        DOCKERFILE_CUDNN_DSO="RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.${CUDNN_MAJOR} /usr/lib/x86_64-linux-gnu/libcudnn.so"
        DOCKERFILE_CUBLAS_DSO="RUN ln -sf /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.${CUDA_MAJOR} /usr/lib/x86_64-linux-gnu/libcublas.so"
    fi

    cat <<EOF
    ========================================
    Generate ${PADDLE_ROOT}/build/Dockerfile ...
    ========================================
EOF

    cat > ${PADDLE_ROOT}/build/Dockerfile <<EOF
    FROM ${BASE_IMAGE}
    MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>
    ENV HOME /root
EOF

    if [[ ${WITH_GPU} == "ON"  ]]; then
        NCCL_DEPS="apt-get install -y --allow-downgrades libnccl2=2.2.13-1+cuda${CUDA_MAJOR} libnccl-dev=2.2.13-1+cuda${CUDA_MAJOR} || true"
    else
        NCCL_DEPS="true"
    fi

    if [[ ${WITH_FLUID_ONLY:-OFF} == "OFF" ]]; then
        PADDLE_VERSION="paddle version"
        CMD='"paddle", "version"'
    else
        PADDLE_VERSION="true"
        CMD='"true"'
    fi

    cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    ADD python/dist/*.whl /
    # run paddle version to install python packages first
    RUN apt-get update && ${NCCL_DEPS}
    RUN apt-get install -y wget python-pip python-opencv libgtk2.0-dev dmidecode python-tk && easy_install -U pip && \
        pip install /*.whl; apt-get install -f -y && \
        apt-get clean -y && \
        rm -f /*.whl && \
        ${PADDLE_VERSION} && \
        ldconfig
    ${DOCKERFILE_CUDNN_DSO}
    ${DOCKERFILE_CUBLAS_DSO}
    ${DOCKERFILE_GPU_ENV}
    ENV NCCL_LAUNCH_MODE PARALLEL
EOF
    if [[ ${WITH_GOLANG:-OFF} == "ON" ]]; then
        cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
        ADD go/cmd/pserver/pserver /usr/bin/
        ADD go/cmd/master/master /usr/bin/
EOF
    fi
    cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    # default command shows the paddle version and exit
    CMD [${CMD}]
EOF
}

function gen_capi_package() {
    if [[ ${WITH_C_API} == "ON" ]]; then
        capi_install_prefix=${INSTALL_PREFIX:-/paddle/build}/capi_output
        rm -rf $capi_install_prefix
        make DESTDIR="$capi_install_prefix" install
        cd $capi_install_prefix/
        ls | egrep -v "^Found.*item$" | xargs tar -czf ${PADDLE_ROOT}/build/paddle.tgz
    fi
}

function gen_fluid_lib() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [[ ${WITH_C_API:-OFF} == "OFF" ]] ; then
        cat <<EOF
    ========================================
    Generating fluid library for train and inference ...
    ========================================
EOF
        cmake .. -DWITH_DISTRIBUTE=OFF -DON_INFER=ON
        make -j `nproc` fluid_lib_dist
        make -j `nproc` inference_lib_dist
      fi
}

function tar_fluid_lib() {
    if [[ ${WITH_C_API:-OFF} == "OFF" ]] ; then
        cat <<EOF
    ========================================
    Taring fluid library for train and inference ...
    ========================================
EOF
        cd ${PADDLE_ROOT}/build
        cp -r fluid_install_dir fluid
        tar -czf fluid.tgz fluid
        cp -r fluid_inference_install_dir fluid_inference
        tar -czf fluid_inference.tgz fluid_inference
      fi
}

function test_fluid_lib() {
    if [[ ${WITH_C_API:-OFF} == "OFF" ]] ; then
        cat <<EOF
    ========================================
    Testing fluid library for inference ...
    ========================================
EOF
        cd ${PADDLE_ROOT}/paddle/fluid/inference/api/demo_ci
        ./run.sh ${PADDLE_ROOT} ${WITH_MKL:-ON} ${WITH_GPU:-OFF} ${INFERENCE_DEMO_INSTALL_DIR} \
                 ${TENSORRT_INCLUDE_DIR:-/usr/local/TensorRT/include} \
                 ${TENSORRT_LIB_DIR:-/usr/local/TensorRT/lib}
        ./clean.sh
      fi
}

function main() {
    local CMD=$1
    init
    case $CMD in
      build)
        cmake_gen ${PYTHON_ABI:-""}
        build
        gen_dockerfile
        ;;
      build_android)
        build_android
        ;;
      build_ios)
        build_ios
        ;;
      test)
        run_test
        ;;
      single_test)
        single_test $2
        ;;
      bind_test)
        bind_test
        ;;
      doc)
        gen_docs
        ;;
      gen_doc_lib)
        gen_doc_lib $2
        ;;
      html)
        gen_html
        ;;
      dockerfile)
        gen_dockerfile
        ;;
      capi)
        cmake_gen ${PYTHON_ABI:-""}
        build
        gen_capi_package
        ;;
      fluid_inference_lib)
        cmake_gen ${PYTHON_ABI:-""}
        gen_fluid_lib
        tar_fluid_lib
        test_fluid_lib
        ;;
      check_style)
        check_style
        ;;
      cicheck)
        cmake_gen ${PYTHON_ABI:-""}
        build
        assert_api_not_changed ${PYTHON_ABI:-""}
        run_test
        gen_capi_package
        gen_fluid_lib
        test_fluid_lib
        assert_api_spec_approvals
        ;;
      maccheck)
        cmake_gen ${PYTHON_ABI:-""}
        build_mac
        run_mac_test ${PROC_RUN:-1}
        ;;
      macbuild)
        cmake_gen ${PYTHON_ABI:-""}
        build_mac
        ;;
      cicheck_py35)
        cmake_gen ${PYTHON_ABI:-""}
        build
        run_test
        assert_api_not_changed ${PYTHON_ABI:-""}
        ;;
      *)
        print_usage
        exit 0
        ;;
      esac
}

main $@
