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
    ${BLUE}test${NONE}: run all unit tests
    ${BLUE}single_test${NONE}: run a single unit test
    ${BLUE}bind_test${NONE}: parallel tests bind to different GPU
    ${BLUE}doc${NONE}: generate paddle documents
    ${BLUE}gen_doc_lib${NONE}: generate paddle documents library
    ${BLUE}html${NONE}: convert C++ source code into HTML
    ${BLUE}dockerfile${NONE}: generate paddle release dockerfile
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
            pip install --user -r ${PADDLE_ROOT}/python/requirements.txt
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
                pip3.5 uninstall -y protobuf
                pip3.5 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp36-cp36m" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.6" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.6/lib/
                export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.6/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.6/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.6/include/python3.6m/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.6/lib/libpython3.6m.dylib"
                pip3.6 uninstall -y protobuf
                pip3.6 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp37-cp37m" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.7" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.7/lib/
                export DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.7/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.7/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.7/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/Library/Frameworks/Python.framework/Versions/3.7/include/python3.7m/
            -DPYTHON_LIBRARY:FILEPATH=/Library/Frameworks/Python.framework/Versions/3.7/lib/libpython3.7m.dylib"
                pip3.7 uninstall -y protobuf
                pip3.7 install --user -r ${PADDLE_ROOT}/python/requirements.txt
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
                pip uninstall -y protobuf
                pip install -r ${PADDLE_ROOT}/python/requirements.txt
            elif [ "$1" == "cp27-cp27mu" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-2.7.11-ucs4/lib:${LD_LIBRARY_PATH#/opt/_internal/cpython-2.7.11-ucs2/lib:}
                export PATH=/opt/python/cp27-cp27mu/bin/:${PATH}
                PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/python/cp27-cp27mu/bin/python
            -DPYTHON_INCLUDE_DIR:PATH=/opt/python/cp27-cp27mu/include/python2.7
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-2.7.11-ucs4/lib/libpython2.7.so"
                pip uninstall -y protobuf
                pip install -r ${PADDLE_ROOT}/python/requirements.txt
            elif [ "$1" == "cp35-cp35m" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.5.1/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.5.1/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.5.1/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.5.1/include/python3.5m
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.5.1/lib/libpython3.so"
                pip3.5 uninstall -y protobuf
                pip3.5 install -r ${PADDLE_ROOT}/python/requirements.txt
            elif [ "$1" == "cp36-cp36m" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.6.0/bin/python3
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.6.0/include/python3.6m
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.6.0/lib/libpython3.so"
                pip3.6 uninstall -y protobuf
                pip3.6 install -r ${PADDLE_ROOT}/python/requirements.txt
            elif [ "$1" == "cp37-cp37m" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
                export PYTHON_FLAGS="-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.7.0/bin/python3.7
            -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.7.0/include/python3.7m
            -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.7.0/lib/libpython3.so"
                pip3.7 uninstall -y protobuf
                pip3.7 install -r ${PADDLE_ROOT}/python/requirements.txt
           fi
        else
            pip uninstall -y protobuf
            pip install -r ${PADDLE_ROOT}/python/requirements.txt
        fi
    fi

    if [ "$SYSTEM" == "Darwin" ]; then
        WITH_DISTRIBUTE=${WITH_DISTRIBUTE:-ON}
        WITH_AVX=${WITH_AVX:-ON}
        INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR:-~/.cache/inference_demo}
    else
        INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR:-/root/.cache/inference_demo}
    fi

    distibuted_flag=${WITH_DISTRIBUTE:-OFF}
    grpc_flag=${WITH_GRPC:-${distibuted_flag}}

    cat <<EOF
    ========================================
    Configuring cmake in /paddle/build ...
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
        ${PYTHON_FLAGS}
        -DWITH_DSO=ON
        -DWITH_GPU=${WITH_GPU:-OFF}
        -DWITH_AMD_GPU=${WITH_AMD_GPU:-OFF}
        -DWITH_DISTRIBUTE=${distibuted_flag}
        -DWITH_MKL=${WITH_MKL:-ON}
        -DWITH_NGRAPH=${WITH_NGRAPH:-OFF}
        -DWITH_AVX=${WITH_AVX:-OFF}
        -DWITH_GOLANG=${WITH_GOLANG:-OFF}
        -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All}
        -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
        -DWITH_PYTHON=${WITH_PYTHON:-ON}
        -DCUDNN_ROOT=/usr/
        -DWITH_TESTING=${WITH_TESTING:-ON}
        -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DWITH_CONTRIB=${WITH_CONTRIB:-ON}
        -DWITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON}
        -DWITH_HIGH_LEVEL_API_TEST=${WITH_HIGH_LEVEL_API_TEST:-OFF}
        -DINFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR}
        -DWITH_ANAKIN=${WITH_ANAKIN:-OFF}
        -DANAKIN_BUILD_FAT_BIN=${ANAKIN_BUILD_FAT_BIN:OFF}
        -DANAKIN_BUILD_CROSS_PLANTFORM=${ANAKIN_BUILD_CROSS_PLANTFORM:ON}
        -DPY_VERSION=${PY_VERSION:-2.7}
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build}
        -DWITH_JEMALLOC=${WITH_JEMALLOC:-OFF} 
        -DWITH_GRPC=${grpc_flag}
    ========================================
EOF
    # Disable UNITTEST_USE_VIRTUALENV in docker because
    # docker environment is fully controlled by this script.
    # See /Paddle/CMakeLists.txt, UNITTEST_USE_VIRTUALENV option.
    cmake .. \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} \
        ${PYTHON_FLAGS} \
        -DWITH_DSO=ON \
        -DWITH_GPU=${WITH_GPU:-OFF} \
        -DWITH_AMD_GPU=${WITH_AMD_GPU:-OFF} \
        -DWITH_DISTRIBUTE=${distibuted_flag} \
        -DWITH_MKL=${WITH_MKL:-ON} \
        -DWITH_NGRAPH=${WITH_NGRAPH:-OFF} \
        -DWITH_AVX=${WITH_AVX:-OFF} \
        -DWITH_GOLANG=${WITH_GOLANG:-OFF} \
        -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All} \
        -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
        -DWITH_PYTHON=${WITH_PYTHON:-ON} \
        -DCUDNN_ROOT=/usr/ \
        -DWITH_TESTING=${WITH_TESTING:-ON} \
        -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DWITH_CONTRIB=${WITH_CONTRIB:-ON} \
        -DWITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON} \
        -DWITH_HIGH_LEVEL_API_TEST=${WITH_HIGH_LEVEL_API_TEST:-OFF} \
        -DINFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR} \
        -DWITH_ANAKIN=${WITH_ANAKIN:-OFF} \
        -DANAKIN_BUILD_FAT_BIN=${ANAKIN_BUILD_FAT_BIN:OFF}\
        -DANAKIN_BUILD_CROSS_PLANTFORM=${ANAKIN_BUILD_CROSS_PLANTFORM:ON}\
        -DPY_VERSION=${PY_VERSION:-2.7} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build} \
        -DWITH_JEMALLOC=${WITH_JEMALLOC:-OFF} \
        -DWITH_GRPC=${grpc_flag}

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

    pip install cpplint
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
    parallel_number=`nproc`
    if [[ "$1" != "" ]]; then
      parallel_number=$1
    fi
    make clean
    make -j ${parallel_number}
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

function run_test() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests ...
    ========================================
EOF
        if [ ${TESTING_DEBUG_MODE:-OFF} == "ON" ] ; then
            ctest -V
        else
            ctest --output-on-failure
        fi
    fi
}

function run_brpc_test() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [[ ${WITH_TESTING:-ON} == "ON" \
        && ${WITH_DISTRIBUTE:-OFF} == "ON" \
        && ${WITH_GRPC:-OFF} == "OFF" ]] ; then
    cat <<EOF
    ========================================
    Running brpc unit tests ...
    ========================================
EOF
        set +x
        declare -a other_tests=("test_listen_and_serv_op" "system_allocator_test" \
        "rpc_server_test" "varhandle_test" "collective_server_test" "brpc_serde_test")
        all_tests=`ctest -N`

        for t in "${other_tests[@]}"
        do
            if [[ ${all_tests} != *$t* ]]; then
                continue
            fi

            if [[ ${TESTING_DEBUG_MODE:-OFF} == "ON" ]] ; then
                ctest -V -R $t
            else
                ctest --output-on-failure -R $t
            fi
        done
        set -x

        if [[ ${TESTING_DEBUG_MODE:-OFF} == "ON" ]] ; then
            ctest -V -R test_dist_*
        else
            ctest --output-on-failure -R test_dist_*
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
        ctest --output-on-failure -j $2
        # make install should also be test when unittest
        make install -j 8
        if [ "$1" == "cp27-cp27m" ]; then
            set -e
            pip install --user ${INSTALL_PREFIX:-/paddle/build}/opt/paddle/share/wheels/*.whl
            python ${PADDLE_ROOT}/paddle/scripts/installation_validate.py
        elif [ "$1" == "cp35-cp35m" ]; then
            pip3.5 install --user ${INSTALL_PREFIX:-/paddle/build}/opt/paddle/share/wheels/*.whl
        elif [ "$1" == "cp36-cp36m" ]; then
            pip3.6 install --user ${INSTALL_PREFIX:-/paddle/build}/opt/paddle/share/wheels/*.whl
        elif [ "$1" == "cp37-cp37m" ]; then
            pip3.7 install --user ${INSTALL_PREFIX:-/paddle/build}/opt/paddle/share/wheels/*.whl
        fi

        paddle version

        if [ "$1" == "cp27-cp27m" ]; then
            pip uninstall -y paddlepaddle
        elif [ "$1" == "cp35-cp35m" ]; then
            pip3.5 uninstall -y paddlepaddle
        elif [ "$1" == "cp36-cp36m" ]; then
            pip3.6 uninstall -y paddlepaddle
        elif [ "$1" == "cp37-cp37m" ]; then
            pip3.7 uninstall -y paddlepaddle
        fi
    fi
}

function assert_api_not_changed() {
    mkdir -p ${PADDLE_ROOT}/build/.check_api_workspace
    cd ${PADDLE_ROOT}/build/.check_api_workspace
    virtualenv .env
    source .env/bin/activate
    pip install ${PADDLE_ROOT}/build/python/dist/*whl
    python ${PADDLE_ROOT}/tools/print_signatures.py paddle.fluid,paddle.reader > new.spec

    if [ "$1" == "cp35-cp35m" ] || [ "$1" == "cp36-cp36m" ] || [ "$1" == "cp37-cp37m" ]; then
        # Use sed to make python2 and python3 sepc keeps the same
        sed -i 's/arg0: str/arg0: unicode/g' new.spec
        sed -i "s/\(.*Transpiler.*\).__init__ (ArgSpec(args=\['self'].*/\1.__init__ /g" new.spec
    fi
    # ComposeNotAligned has significant difference between py2 and py3
    sed -i '/.*ComposeNotAligned.*/d' new.spec

    python ${PADDLE_ROOT}/tools/diff_api.py ${PADDLE_ROOT}/paddle/fluid/API.spec new.spec

    # Currently, we only check in PR_CI python 2.7
    if [ "$SYSTEM" != "Darwin" ]; then
      if [ "$1" == "" ] || [ "$1" == "cp27-cp27m" ] || [ "$1" == "cp27-cp27mu" ]; then
        python ${PADDLE_ROOT}/tools/diff_use_default_grad_op_maker.py ${PADDLE_ROOT}/paddle/fluid/op_use_default_grad_op_maker.spec
      fi
    fi
    deactivate
}

function assert_api_spec_approvals() {
    if [ -z ${BRANCH} ]; then
        BRANCH="develop"
    fi

    API_FILES=("CMakeLists.txt"
               "paddle/fluid/API.spec"
               "paddle/fluid/op_use_default_grad_op_maker.spec"
               "python/paddle/fluid/parallel_executor.py"
               "paddle/fluid/framework/operator.h"
               "paddle/fluid/framework/tensor.h"
               "paddle/fluid/framework/details/op_registry.h"
               "paddle/fluid/framework/grad_op_desc_maker.h"
               "paddle/fluid/framework/lod_tensor.h"
               "paddle/fluid/framework/selected_rows.h"
               "paddle/fluid/framework/op_desc.h"
               "paddle/fluid/framework/block_desc.h"
               "paddle/fluid/framework/var_desc.h"
               "paddle/fluid/framework/scope.h"
               "paddle/fluid/framework/ir/node.h"
               "paddle/fluid/framework/ir/graph.h"
               "paddle/fluid/framework/framework.proto"
               "python/paddle/fluid/compiler.py"
               "paddle/fluid/operators/distributed/send_recv.proto.in")
    for API_FILE in ${API_FILES[*]}; do
      API_CHANGE=`git diff --name-only upstream/$BRANCH | grep "${API_FILE}" || true`
      echo "checking ${API_FILE} change, PR: ${GIT_PR_ID}, changes: ${API_CHANGE}"
      if [ ${API_CHANGE} ] && [ "${GIT_PR_ID}" != "" ]; then
          # NOTE: per_page=10000 should be ok for all cases, a PR review > 10000 is not human readable.
          # approval_user_list: velconia 1979255,XiaoguangHu01 46782768,chengduoZH 30176695,Xreki 12538138,luotao1 6836917,sneaxiy 32832641,tensor-tang 21351065,jacquesqiao 3048612,typhoonzero 13348433,shanyi15 35982308. 
          if [ "$API_FILE" == "paddle/fluid/API.spec" ];then
            APPROVALS=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000 | \
            python ${PADDLE_ROOT}/tools/check_pr_approval.py 2 35982308 46782768 30176695`
            if [ "${APPROVALS}" == "TRUE" ];then
              APPROVALS=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000 | \
              python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 35982308`
            fi
          elif [ "$API_FILE" == "CMakeLists.txt" ];then
            APPROVALS=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000 | \
            python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 6836917 46782768 30176695`
          else
            APPROVALS=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000 | \
            python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 1979255 21351065 3048612 13348433 46782768 30176695 12538138 6836917 32832641`
          fi
          echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
          if [ "${APPROVALS}" == "FALSE" ]; then
            if [ "$API_FILE" == "paddle/fluid/API.spec" ];then
              echo "You must have one RD (chengduoZH or XiaoguangHu01) and one PM (shanyi15) approval for the api change! ${API_FILE}"
            elif [ "$API_FILE" == "CMakeLists.txt" ];then
              echo "You must have one RD (luotao1 or chengduoZH or XiaoguangHu01) approval for the cmakelist change! ${API_FILE}"
            else
              echo "You must have one RD (velconia,XiaoguangHu01,chengduoZH,Xreki,luotao1,sneaxiy,tensor-tang,jacquesqiao,typhoonzero) approval for the api change! ${API_FILE}"
            fi
            exit 1
          fi
      fi
    done

    HAS_CONST_CAST=`git diff -U0 upstream/$BRANCH |grep -o -m 1 "const_cast" || true`
    if [ ${HAS_CONST_CAST} ] && [ "${GIT_PR_ID}" != "" ]; then
        APPROVALS=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000 | \
        python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 1979255 21351065 3048612 13348433 46782768 30176695 12538138 6836917 32832641`
        echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
        if [ "${APPROVALS}" == "FALSE" ]; then
            echo "You must have one RD (velconia,XiaoguangHu01,chengduoZH,Xreki,luotao1,sneaxiy,tensor-tang,jacquesqiao,typhoonzero) approval for the api change! ${API_FILE}"
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

function parallel_test() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests ...
    ========================================
EOF

        # calculate and set the memory usage for each process
        # MEM_USAGE=$(printf "%.2f" `echo "scale=5; 1.0 / $NUM_PROC" | bc`)
        # export FLAGS_fraction_of_gpu_memory_to_use=$MEM_USAGE

        EXIT_CODE=0;
        pids=()

        # get the CUDA device count
        CUDA_DEVICE_COUNT=$(nvidia-smi -L | wc -l)
        # each test case would occupy two graph cards
        NUM_PROC=$[CUDA_DEVICE_COUNT/2]
        for (( i = 0; i < $NUM_PROC; i++ )); do
            # CUDA_VISIBLE_DEVICES http://acceleware.com/blog/cudavisibledevices-masking-gpus
            # ctest -I https://cmake.org/cmake/help/v3.0/manual/ctest.1.html?highlight=ctest
            if [ ${TESTING_DEBUG_MODE:-OFF} == "ON" ] ; then
                env CUDA_VISIBLE_DEVICES=$[i*2],$[i*2+1] ctest -I $i,,$NUM_PROC -V &
                pids+=($!)
            else
                env CUDA_VISIBLE_DEVICES=$[i*2],$[i*2+1] ctest -I $i,,$NUM_PROC --output-on-failure &
                pids+=($!)
            fi
        done

        clen=`expr "${#pids[@]}" - 1` # get length of commands - 1
        for i in `seq 0 "$clen"`; do
            wait ${pids[$i]}
            CODE=$?
            if [[ "${CODE}" != "0" ]]; then
                echo "At least one test failed with exit code => ${CODE}" ;
                EXIT_CODE=1;
            fi
        done
        wait; # wait for all subshells to finish

        echo "EXIT_CODE => $EXIT_CODE"
        if [[ "${EXIT_CODE}" != "0" ]]; then
            exit "$EXIT_CODE"
        fi
    fi
}

function card_test() {
    # get the CUDA device count
    CUDA_DEVICE_COUNT=$(nvidia-smi -L | wc -l)

    testcases=$1
    if (( $# > 1 )); then
        cardnumber=$2
        if (( $cardnumber > $CUDA_DEVICE_COUNT )); then
            cardnumber=$CUDA_DEVICE_COUNT
        fi
    else
        cardnumber=$CUDA_DEVICE_COUNT
    fi

    if [[ "$testcases" == "" ]]; then
        return 0
    fi

    EXIT_CODE=0;
    pids=()

    NUM_PROC=$[CUDA_DEVICE_COUNT/$cardnumber]
    for (( i = 0; i < $NUM_PROC; i++ )); do
        # CUDA_VISIBLE_DEVICES http://acceleware.com/blog/cudavisibledevices-masking-gpus
        # ctest -I https://cmake.org/cmake/help/v3.0/manual/ctest.1.html?highlight=ctest
        cuda_list=()
        for (( j = 0; j < cardnumber; j++ )); do
            if [ $j -eq 0 ]; then
                    cuda_list=("$[i*cardnumber]")
                else
                    cuda_list="$cuda_list,$[i*cardnumber+j]"
            fi
        done
        # echo $cuda_list
        if [ ${TESTING_DEBUG_MODE:-OFF} == "ON" ] ; then
            env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -V &
            pids+=($!)
        else
#            echo "env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R ($testcases) --output-on-failure &"
            env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" --output-on-failure &
            pids+=($!)
        fi
    done

    clen=`expr "${#pids[@]}" - 1` # get length of commands - 1
    for i in `seq 0 "$clen"`; do
        wait ${pids[$i]}
        CODE=$?
        if [[ "${CODE}" != "0" ]]; then
            echo "At least one test failed with exit code => ${CODE}" ;
            EXIT_CODE=1;
        fi
    done
    wait; # wait for all subshells to finish

    echo "EXIT_CODE => $EXIT_CODE"
    return $EXIT_CODE
}

function aggresive_test() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    if [ ${WITH_TESTING:-ON} == "ON" ] ; then
    cat <<EOF
    ========================================
    Running unit tests ...
    ========================================
EOF

        EXIT_CODE=0;
        test_cases=$(ctest -N -V)
        exclusive_tests=''
        single_card_tests=''
        multiple_card_tests=''
        is_exclusive=''
        is_multicard=''
        while read -r line; do
            if [[ "$line" == "" ]]; then
                continue
            fi
                read matchstr <<< $(echo "$line"|grep -oEi 'Test[ \t]+#')
                if [[ "$matchstr" == "" ]]; then
                    # Any test case with LABELS property would be parse here
                    # RUN_TYPE=EXCLUSIVE mean the case would run exclusively
                    # RUN_TYPE=DIST mean the case would take two graph cards during runtime
                    read is_exclusive <<< $(echo "$line"|grep -oEi "RUN_TYPE=EXCLUSIVE")
                    read is_multicard <<< $(echo "$line"|grep -oEi "RUN_TYPE=DIST")
                    continue
                fi
                read testcase <<< $(echo "$line"|grep -oEi "\w+$")

                if [[ "$is_multicard" == "" ]]; then
                  # trick: treat all test case with prefix "test_dist" as dist case, and would run on 2 cards
                  read is_multicard <<< $(echo "$testcase"|grep -oEi "test_dist")
                fi

                if [[ "$is_exclusive" != "" ]]; then
                    if [[ "$exclusive_tests" == "" ]]; then
                        exclusive_tests=$testcase
                    else
                        exclusive_tests="$exclusive_tests|$testcase"
                    fi
                elif [[ "$is_multicard" != "" ]]; then
                    if [[ "$multiple_card_tests" == "" ]]; then
                        multiple_card_tests=$testcase
                    else
                        multiple_card_tests="$multiple_card_tests|$testcase"
                    fi
                else
                    if [[ "$single_card_tests" == "" ]]; then
                        single_card_tests=$testcase
                    else
                        single_card_tests="$single_card_tests|$testcase"
                    fi
                fi
                is_exclusive=''
                is_multicard=''
                matchstr=''
                testcase=''
        done <<< "$test_cases";

        card_test "$single_card_tests" 1
        if [[ "$?" != "0" ]]; then
            EXIT_CODE=1
        fi
        card_test "$multiple_card_tests" 2
        if [[ "$?" != "0" ]]; then
            EXIT_CODE=1
        fi
        card_test "$exclusive_tests"
        if [[ "$?" != "0" ]]; then
            EXIT_CODE=1
        fi
        if [[ "$EXIT_CODE" != "0" ]]; then
            exit 1;
        fi
    fi
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
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \

    local LIB_TYPE=$1
    case $LIB_TYPE in
      full)
        # Build full Paddle Python module. Will timeout without caching 'copy_paddle_pybind' first
        make -j `nproc` framework_py_proto copy_paddle_pybind paddle_python
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

    PADDLE_VERSION="paddle version"
    CMD='"paddle", "version"'

    if [ "$1" == "cp35-cp35m" ]; then
        cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    ADD python/dist/*.whl /
    # run paddle version to install python packages first
    RUN apt-get update && ${NCCL_DEPS}
    RUN apt-get install -y wget python3 python3-pip libgtk2.0-dev dmidecode python3-tk && \
        pip3 install opencv-python && pip3 install /*.whl; apt-get install -f -y && \
        apt-get clean -y && \
        rm -f /*.whl && \
        ${PADDLE_VERSION} && \
        ldconfig
    ${DOCKERFILE_CUDNN_DSO}
    ${DOCKERFILE_CUBLAS_DSO}
    ${DOCKERFILE_GPU_ENV}
    ENV NCCL_LAUNCH_MODE PARALLEL
EOF
    elif [ "$1" == "cp36-cp36m" ]; then
        cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    ADD python/dist/*.whl /
    # run paddle version to install python packages first
    RUN apt-get update && ${NCCL_DEPS}
    RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev
    RUN mkdir -p /root/python_build/ && wget -q https://www.sqlite.org/2018/sqlite-autoconf-3250300.tar.gz && \
        tar -zxf sqlite-autoconf-3250300.tar.gz && cd sqlite-autoconf-3250300 && \
        ./configure -prefix=/usr/local && make -j8 && make install && cd ../ && rm sqlite-autoconf-3250300.tar.gz && \
        wget -q https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tgz && \
        tar -xzf Python-3.6.0.tgz && cd Python-3.6.0 && \
        CFLAGS="-Wformat" ./configure --prefix=/usr/local/ --enable-shared > /dev/null && \
        make -j8 > /dev/null && make altinstall > /dev/null
    RUN apt-get install -y libgtk2.0-dev dmidecode python3-tk && \
        pip3.6 install opencv-python && pip3.6 install /*.whl; apt-get install -f -y && \
        apt-get clean -y && \
        rm -f /*.whl && \
        ${PADDLE_VERSION} && \
        ldconfig
    ${DOCKERFILE_CUDNN_DSO}
    ${DOCKERFILE_CUBLAS_DSO}
    ${DOCKERFILE_GPU_ENV}
    ENV NCCL_LAUNCH_MODE PARALLEL
EOF
    elif [ "$1" == "cp37-cp37m" ]; then
        cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    ADD python/dist/*.whl /
    # run paddle version to install python packages first
    RUN apt-get update && ${NCCL_DEPS}
    RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev
    RUN wget -q https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz && \
        tar -xzf Python-3.7.0.tgz && cd Python-3.7.0 && \
        CFLAGS="-Wformat" ./configure --prefix=/usr/local/ --enable-shared > /dev/null && \
        make -j8 > /dev/null && make altinstall > /dev/null
    RUN apt-get install -y libgtk2.0-dev dmidecode python3-tk && \
        pip3.7 install opencv-python && pip3.7 install /*.whl; apt-get install -f -y && \
        apt-get clean -y && \
        rm -f /*.whl && \
        ${PADDLE_VERSION} && \
        ldconfig
    ${DOCKERFILE_CUDNN_DSO}
    ${DOCKERFILE_CUBLAS_DSO}
    ${DOCKERFILE_GPU_ENV}
    ENV NCCL_LAUNCH_MODE PARALLEL
EOF
    else
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
    fi

    cat >> ${PADDLE_ROOT}/build/Dockerfile <<EOF
    # default command shows the paddle version and exit
    CMD [${CMD}]
EOF
}

function gen_fluid_lib() {
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    cat <<EOF
    ========================================
    Generating fluid library for train and inference ...
    ========================================
EOF
    parallel_number=`nproc`
    if [[ "$1" != "" ]]; then
      parallel_number=$1
    fi
    cmake .. -DWITH_DISTRIBUTE=OFF -DON_INFER=ON -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All} -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN}

    make -j ${parallel_number} fluid_lib_dist
    make -j ${parallel_number} inference_lib_dist
}

function tar_fluid_lib() {
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
}

function test_fluid_lib() {
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
}

function main() {
    local CMD=$1
    local parallel_number=$2
    init
    case $CMD in
      build_only)
        cmake_gen ${PYTHON_ABI:-""}
        build ${parallel_number}
        ;;
      build_and_check)
        cmake_gen ${PYTHON_ABI:-""}
        build ${parallel_number}
        assert_api_not_changed ${PYTHON_ABI:-""}
        assert_api_spec_approvals
        ;;
      build)
#        cmake_gen ${PYTHON_ABI:-""}
#        build ${parallel_number}
#        gen_dockerfile ${PYTHON_ABI:-""}
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
      gen_doc_lib)
        gen_doc_lib $2
        ;;
      html)
        gen_html
        ;;
      dockerfile)
        gen_dockerfile ${PYTHON_ABI:-""}
        ;;
      fluid_inference_lib)
        cmake_gen ${PYTHON_ABI:-""}
        gen_fluid_lib ${parallel_number}
        tar_fluid_lib
        test_fluid_lib
        ;;
      check_style)
        check_style
        ;;
      cicheck)
        cmake_gen ${PYTHON_ABI:-""}
        build ${parallel_number}
#        assert_api_not_changed ${PYTHON_ABI:-""}
        aggresive_test
#        gen_fluid_lib ${parallel_number}
#        test_fluid_lib
#        assert_api_spec_approvals
        ;;
      cicheck_brpc)
        cmake_gen ${PYTHON_ABI:-""}
        build ${parallel_number}
        run_brpc_test
        ;;
      assert_api)
        assert_api_not_changed ${PYTHON_ABI:-""}
        assert_api_spec_approvals
        ;;
      test_inference)
        gen_fluid_lib ${parallel_number}
        test_fluid_lib
        ;;
      assert_api_approvals)
        assert_api_spec_approvals
        ;;
      maccheck)
        cmake_gen ${PYTHON_ABI:-""}
        build_mac
        run_mac_test ${PYTHON_ABI:-""} ${PROC_RUN:-1}
        ;;
      macbuild)
        cmake_gen ${PYTHON_ABI:-""}
        build_mac
        ;;
      cicheck_py35)
        cmake_gen ${PYTHON_ABI:-""}
        build ${parallel_number}
        aggresive_test
        assert_api_not_changed ${PYTHON_ABI:-""}
        ;;
      cmake_gen)
        cmake_gen ${PYTHON_ABI:-""}
        ;;
      gen_fluid_lib)
        gen_fluid_lib ${parallel_number}
        ;;
      test_fluid_lib)
        test_fluid_lib
        ;;
      *)
        print_usage
        exit 1
        ;;
      esac
}

main $@
