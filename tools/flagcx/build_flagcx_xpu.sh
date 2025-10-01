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

# Positional arguments
XPU_INCLUDE_PATH="$1"   # e.g. /workspace/Paddle/build/third_party/install/xpu/include/xpu
XPU_LIB_PATH="$2"       # e.g. /workspace/Paddle/build/third_party/install/xpu/lib
FLAGCX_SOURCE_PATH="$3" # e.g. /workspace/Paddle/third_party/flagcx/

# Ensure /usr/local/xccl exists
if [ ! -d "/usr/local/xccl" ]; then
    echo "[INFO] Creating /usr/local/xccl"
    sudo mkdir -p /usr/local/xccl
fi

# Ensure /usr/local/xccl/include symlink exists
if [ ! -L "/usr/local/xccl/include" ]; then
    echo "[INFO] Creating symlink for include directory"
    sudo ln -s "${XPU_INCLUDE_PATH}" /usr/local/xccl/include
else
    echo "[INFO] /usr/local/xccl/include already exists — skipping"
fi

# Ensure /usr/local/xccl/so symlink exists
if [ ! -L "/usr/local/xccl/so" ]; then
    echo "[INFO] Creating symlink for lib directory"
    sudo ln -s "${XPU_LIB_PATH}" /usr/local/xccl/so
else
    echo "[INFO] /usr/local/xccl/so already exists — skipping"
fi

cd "${FLAGCX_SOURCE_PATH}"
make -j1 clean
make -j1 USE_KUNLUNXIN=1
