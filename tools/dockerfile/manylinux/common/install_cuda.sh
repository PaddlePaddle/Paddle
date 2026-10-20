#!/bin/bash

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

set -ex

function install_cusparselt_063 {
    # cuSparseLt license: https://docs.nvidia.com/cuda/cusparselt/license.html
    mkdir tmp_cusparselt && pushd tmp_cusparselt
    wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.6.3.2-archive.tar.xz
    tar xf libcusparse_lt-linux-x86_64-0.6.3.2-archive.tar.xz
    cp -a libcusparse_lt-linux-x86_64-0.6.3.2-archive/include/* /usr/local/cuda/include/
    cp -a libcusparse_lt-linux-x86_64-0.6.3.2-archive/lib/* /usr/local/cuda/lib64/
    popd
    rm -rf tmp_cusparselt
}

function install_cusparselt_090_cuda13 {
    # cuSparseLt license: https://docs.nvidia.com/cuda/cusparselt/license.html
    mkdir tmp_cusparselt && pushd tmp_cusparselt
    wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.9.0.3_cuda13-archive.tar.xz
    tar xf libcusparse_lt-linux-x86_64-0.9.0.3_cuda13-archive.tar.xz
    cp -a libcusparse_lt-linux-x86_64-0.9.0.3_cuda13-archive/include/* /usr/local/cuda/include/
    cp -a libcusparse_lt-linux-x86_64-0.9.0.3_cuda13-archive/lib/* /usr/local/cuda/lib64/
    popd
    rm -rf tmp_cusparselt
}

function install_cusparselt_081 {
    # cuSparseLt license: https://docs.nvidia.com/cuda/cusparselt/license.html
    mkdir tmp_cusparselt && pushd tmp_cusparselt
    wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.8.1.1_cuda13-archive.tar.xz
    tar xf libcusparse_lt-linux-x86_64-0.8.1.1_cuda13-archive.tar.xz
    cp -a libcusparse_lt-linux-x86_64-0.8.1.1_cuda13-archive/include/* /usr/local/cuda/include/
    cp -a libcusparse_lt-linux-x86_64-0.8.1.1_cuda13-archive/lib/* /usr/local/cuda/lib64/
    popd
    rm -rf tmp_cusparselt
}

function install_nccl_2251_cuda128 {
    yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    yum install -y \
        cuda-compat-12-8 \
        libnccl-2.25.1-1+cuda12.8 \
        libnccl-devel-2.25.1-1+cuda12.8 \
        libnccl-static-2.25.1-1+cuda12.8
}

function install_nccl_2234 {
    wget -q https://nccl2-deb.cdn.bcebos.com/nccl_2.23.4-1+cuda12.6_x86_64.txz --no-check-certificate --no-proxy
    tar xf nccl_2.23.4-1+cuda12.6_x86_64.txz
    cp -a nccl_2.23.4-1+cuda12.6_x86_64/include/* /usr/include/
    cp -a nccl_2.23.4-1+cuda12.6_x86_64/lib/* /usr/lib64
    rm -rf nccl_2.23.4-1+cuda12.6_x86_64 nccl_2.23.4-1+cuda12.6_x86_64.txz
}

function install_nccl_2297_cuda132 {
    yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    # `install_132` uses toolkit-only installation, so install the official
    # forward-compat driver libraries to provide /usr/local/cuda-13.2/compat/libcuda.so.1.
    yum install -y \
        cuda-compat-13-2 \
        libnccl-2.29.7-1+cuda13.2 \
        libnccl-devel-2.29.7-1+cuda13.2 \
        libnccl-static-2.29.7-1+cuda13.2
}

function install_nccl_2283 {
    wget -q https://nccl2-deb.cdn.bcebos.com/nccl_2.28.3-1+cuda13.0_x86_64.txz --no-check-certificate --no-proxy
    tar xf nccl_2.28.3-1+cuda13.0_x86_64.txz
    cp -a nccl_2.28.3-1+cuda13.0_x86_64/include/* /usr/include/
    cp -a nccl_2.28.3-1+cuda13.0_x86_64/lib/* /usr/lib64
    rm -rf nccl_2.28.3-1+cuda13.0_x86_64 nccl_2.28.3-1+cuda13.0_x86_64.txz
}

function install_128 {
    CUDNN_VERSION=9.7.1.26
    NCCL_VERSION=2.25.1
    echo "Installing CUDA 12.8.1 and cuDNN ${CUDNN_VERSION} and NCCL ${NCCL_VERSION} and cuSparseLt-0.6.3"
    rm -rf /usr/local/cuda-12.8 /usr/local/cuda
    wget -q https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
    chmod +x cuda_12.8.1_570.124.06_linux.run
    ./cuda_12.8.1_570.124.06_linux.run --toolkit --silent
    rm -f cuda_12.8.1_570.124.06_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.8 /usr/local/cuda

    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn

    install_nccl_2251_cuda128
    install_cusparselt_063
    ldconfig
}

function install_129 {
    CUDNN_VERSION=9.9.0.52
    NCCL_VERSION=2.23.4
    echo "Installing CUDA 12.9.0 and cuDNN ${CUDNN_VERSION} and NCCL ${NCCL_VERSION} and cuSparseLt-0.6.3"
    rm -rf /usr/local/cuda-12.9 /usr/local/cuda
    # install CUDA 12.9.0 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
    chmod +x cuda_12.9.0_575.51.03_linux.run
    ./cuda_12.9.0_575.51.03_linux.run --toolkit --driver --silent --kernel-source-path=/usr/src/kernels/4.18.0-553.34.1.el8_10.x86_64
    rm -f cuda_12.9.0_575.51.03_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.9 /usr/local/cuda
    rm -rf /usr/bin/nvidia-smi

    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn

    install_nccl_2234
    install_cusparselt_063

    ldconfig
}

function install_130 {
    CUDNN_VERSION=9.13.0.50
    NCCL_VERSION=2.28.3
    echo "Installing CUDA 13.0.1 and cuDNN ${CUDNN_VERSION} and NCCL ${NCCL_VERSION} and cuSparseLt-0.8.1"
    rm -rf /usr/local/cuda-13.0 /usr/local/cuda
    # install CUDA 13.0.1 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda_13.0.1_580.82.07_linux.run
    chmod +x cuda_13.0.1_580.82.07_linux.run
    ./cuda_13.0.1_580.82.07_linux.run --toolkit --driver --silent --kernel-source-path=/usr/src/kernels/4.18.0-553.76.1.el8_10.x86_64
    rm -f cuda_13.0.1_580.82.07_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-13.0 /usr/local/cuda
    rm -rf /usr/bin/nvidia-smi

    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn

    install_nccl_2283
    install_cusparselt_081

    ldconfig
}

function install_132 {
    CUDNN_VERSION=9.20.0.48
    NCCL_VERSION=2.29.7
    echo "Installing CUDA 13.2.0 and cuDNN ${CUDNN_VERSION} and NCCL ${NCCL_VERSION} and cuSparseLt-0.9.0"
    rm -rf /usr/local/cuda-13.2 /usr/local/cuda
    # install CUDA 13.2.0 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_595.45.04_linux.run
    chmod +x cuda_13.2.0_595.45.04_linux.run
    ./cuda_13.2.0_595.45.04_linux.run --toolkit --driver --silent --kernel-source-path=/usr/src/kernels/4.18.0-553.34.1.el8_10.x86_64
    rm -f cuda_13.2.0_595.45.04_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-13.2 /usr/local/cuda
    rm -rf /usr/bin/nvidia-smi

    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn

    install_nccl_2297_cuda132
    install_cusparselt_090_cuda13

    ldconfig
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    12.8) install_128
        ;;
    12.9) install_129
        ;;
    13.0) install_130
        ;;
    13.2) install_132
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done
