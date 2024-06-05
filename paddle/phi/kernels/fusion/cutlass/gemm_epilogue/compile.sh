# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
set -e

build_directory="build"
if [ ! -d "$build_directory" ]; then
    mkdir $build_directory
fi

libname="$build_directory/libCutlassGemmEpilogue.so"
if [ -e "$libname" ]; then
    exit 0
fi

default_python_exe_path="/usr/bin/python"
default_cuda_root_path="/usr/local/cuda"
default_gpu_cc="80"
default_cmake_command="cmake"

python_exe_path="${1:-$default_python_exe_path}"
cuda_root_path="${2:-$default_cuda_root_path}"
gpu_cc="${3:-$default_gpu_cc}"
cmake_command="${4:-$default_cmake_command}"

case "$gpu_cc" in
    80|86|89)  ;;
    *)  exit 0  ;;
esac

cutlass_repo_directory="cutlass"
if [ ! -d "$cutlass_repo_directory" ]; then
    git clone --branch v2.11.0  https://github.com/NVIDIA/cutlass
fi


cd $build_directory
$cmake_command .. -DPYTHON_EXECUTABLE=$python_exe_path -DCUDA_TOOLKIT_ROOT_DIR=$cuda_root_path -DCOMPUTE_CAPABILITY=$gpu_cc
make -j8
cd -
