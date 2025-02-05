#!/usr/bin/env bash

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

set -ex
workspace=$(cd $(dirname ${BASH_SOURCE[0]})/../..; pwd)
build_dir_name=${cinn_build:-build}
build_dir=$workspace/${build_dir_name}
py_version=${py_version:-3.10}
cinn_whl_path=python/dist/cinn-0.0.0-py3-none-any.whl


#export LLVM11_DIR=${workspace}/THIRDS/usr

if [[ "" == ${JOBS} ]]; then
  JOBS=`nproc`
fi

cuda_config=OFF
cudnn_config=OFF

mklcblas_config=ON
onednn_config=ON

function mklcblas_off {
  mklcblas_config=OFF
}
function onednn_off {
  onednn_config=OFF
}

set +x
OLD_HTTP_PROXY=$http_proxy &> /dev/null
OLD_HTTPS_PROXY=$https_proxy &> /dev/null
set -x

function proxy_on {
  set +x
  export http_proxy=$OLD_HTTP_PROXY &> /dev/null
  export https_proxy=$OLD_HTTPS_PROXY &> /dev/null
  set -x
}

function prepare_ci {
  cd $workspace
  proxy_on

  if [[ $(command -v python) == $build_dir/ci-env/bin/python ]]; then
    return
  elif [[ -e $build_dir/ci-env/bin/activate ]]; then
    source $build_dir/ci-env/bin/activate
    return
  fi

  echo "the current user EUID=$EUID: $(whoami)"

  if [[ ! -e $build_dir/ci-env/bin/activate ]]; then
    virtualenv ${build_dir}/ci-env -p python${py_version}
  fi

  source $build_dir/ci-env/bin/activate
  python${py_version} -m pip install -U --no-cache-dir pip
  python${py_version} -m pip install wheel
  python${py_version} -m pip install sphinx==3.3.1 sphinx_gallery==0.8.1 recommonmark==0.6.0 exhale scipy breathe==4.24.0 matplotlib sphinx_rtd_theme
  python${py_version} -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
}


function cmake_ {
    mkdir -p $build_dir
    cd $build_dir
    set -x
    cmake ${workspace} -DWITH_CINN=ON -DWITH_GPU=${cuda_config} \
      -DWITH_TESTING=ON  -DWITH_MKL=${mklcblas_config}  -DCINN_WITH_CUDNN=${cudnn_config} \
      -DPY_VERSION=${py_version}
    set +x

}

function _download_and_untar {
    local tar_file=$1
    if [[ ! -f $tar_file ]]; then
        wget -q https://paddle-inference-dist.bj.bcebos.com/CINN/$tar_file
        tar -zxf $tar_file
    fi
}

function prepare_model {
    cd $build_dir/third_party

    _download_and_untar ResNet18.tar.gz
    _download_and_untar MobileNetV2.tar.gz
    _download_and_untar EfficientNet.tar.gz
    _download_and_untar MobilenetV1.tar.gz
    _download_and_untar ResNet50.tar.gz
    _download_and_untar SqueezeNet.tar.gz
    _download_and_untar FaceDet.tar.gz


    mkdir -p $build_dir/third_party/model
    cd $build_dir/third_party/model
    tar_file="lite_naive_model.tar.gz"
    if [[ ! -f $tar_file ]]; then
        wget -q https://paddle-inference-dist.bj.bcebos.com/$tar_file
        tar -zxf $tar_file
    fi

    proxy_on
    mkdir -p $build_dir/paddle
    cd $build_dir/third_party
    python${py_version} $workspace/test/cinn/fake_model/naive_mul.py
    python${py_version} $workspace/test/cinn/fake_model/naive_multi_fc.py
    python${py_version} $workspace/test/cinn/fake_model/resnet_model.py
}

function build {
    proxy_on
    cd $build_dir

    make -j $JOBS

    ls python/dist
    python${py_version} -m pip install xgboost
    python${py_version} -m pip install -U ${cinn_whl_path}
}

function run_demo {
    cd $build_dir/dist
    export runtime_include_dir=$workspace/paddle/cinn/runtime/cuda
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$build_dir/dist/cinn/lib
    bash build_demo.sh
    ./demo
    rm ./demo
    cd -
}

function run_test {
    source $build_dir/ci-env/bin/activate
    cd $build_dir
    export runtime_include_dir=$workspace/paddle/cinn/runtime/cuda

    if [ ${TESTING_DEBUG_MODE:-OFF} == "ON" ] ; then
        ctest --parallel 10 -V -E "test_frontend_interpreter|test_cinn_fake_resnet|test_dce_pass"
    else
        ctest --parallel 10 --output-on-failure -E "test_frontend_interpreter|test_cinn_fake_resnet|test_dce_pass"
    fi
}

function CINNRT {
    mkdir -p $build_dir
    cd $build_dir
    export runtime_include_dir=$workspace/paddle/cinn/runtime/cuda

    prepare_ci

    mkdir -p $build_dir
    cd $build_dir
    set -x
    cmake ${workspace} -DWITH_CINN=ON -DWITH_GPU=${cuda_config} \
      -DWITH_TESTING=ON  -DWITH_MKL=${mklcblas_config} -DPUBLISH_LIBS=ON
    set +x
    make cinnopt -j $JOBS
}

function main {
    # Parse command line.
    for i in "$@"; do
        case $i in
            mklcblas_off)
                mklcblas_off
                onednn_off
                shift
                ;;
            onednn_off)
                onednn_off
                shift
                ;;
            check_style)
                codestyle_check
                shift
                ;;
            cmake)
                cmake_
                shift
                ;;
            build)
                build
                shift
                ;;
            test)
                run_test
                shift
                ;;
            CINNRT)
               CINNRT
               shift
                ;;
            prepare_model)
                prepare_model
                shift
                ;;
        esac
    done
}

main $@
