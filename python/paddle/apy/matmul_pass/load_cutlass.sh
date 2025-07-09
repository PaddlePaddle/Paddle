# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

SOURCE_CUTLASS_DIR="../../../../third_party/cutlass"
SOURCE_CUTLASS_GIT_DIR="../../../../../../.git/modules/third_party/cutlass"
cutlass_repo_directory="matmul/cutlass-3.7.0"
if [ ! -d "$cutlass_repo_directory" ]; then
    if [ -d "$SOURCE_CUTLASS_DIR" ]; then
        echo "Cutlass folder exists in the submodule and is being copied to the current directory..."
        cp -r "$SOURCE_CUTLASS_DIR" "$cutlass_repo_directory"
        cd "$cutlass_repo_directory"
        echo "Copy the .git directory of cutlass to the current directory"
        rm -rf .git
        cp -r "$SOURCE_CUTLASS_GIT_DIR" .git
        sed -i '6c\ \ worktree = ../' .git/config
        git checkout v3.7.0
        cd ..
        mkdir "cutlass"
        cp -r "cutlass-3.7.0/tools" "cutlass"
        cp -r "cutlass-3.7.0/include" "cutlass"
        rm -rf "cutlass-3.7.0"
        mv "cutlass" "cutlass-3.7.0"
    else
        echo "Cutlass folder does not exist in the submodule and is being downloaded..."
        git clone --branch v3.7.0  https://github.com/NVIDIA/cutlass "$cutlass_repo_directory"
    fi
fi
