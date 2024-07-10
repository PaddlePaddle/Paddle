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

libname="$build_directory/libCutlassConv2d.so"
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
    75|80|86|89)  ;;
    *)  exit 0  ;;
esac

SOURCE_CUTLASS_DIR="../../../../../../third_party/cutlass"
SOURCE_CUTLASS_GIT_DIR="../../../../../../../.git/modules/third_party/cutlass"
cutlass_repo_directory="cutlass"
if [ ! -d "$cutlass_repo_directory" ]; then
    if [ -d "$SOURCE_CUTLASS_DIR" ]; then
        echo "Cutlass folder exists in the submodule and is being copied to the current directory..."
        cp -r "$SOURCE_CUTLASS_DIR" cutlass
        cd cutlass
        echo "Copy the .git directory of cutlass to the current directory"
        rm -rf .git
        cp -r "$SOURCE_CUTLASS_GIT_DIR" .git
        sed -i '6c\ \ worktree = ../' .git/config
        git checkout v3.0.0
        cd ..
    else
        echo "Cutlass folder does not exist in the submodule and is being downloaded..."
        git clone --branch v3.0.0  https://github.com/NVIDIA/cutlass
    fi
fi


cd $build_directory
$cmake_command .. -DPYTHON_EXECUTABLE=$python_exe_path -DCUDA_TOOLKIT_ROOT_DIR=$cuda_root_path -DCOMPUTE_CAPABILITY=$gpu_cc
make -j8
cd -
