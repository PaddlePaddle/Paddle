#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

set -u

if [[ $# -ne 2 ]]; then
    echo "usage: ./check_xpu_dependence.sh XPU_BASE_URL XPU_XCCL_BASE_URL"
    exit 1
fi

xpu_base_url=$1
xccl_base_url=$2

echo "xpu_base_url: $xpu_base_url"
echo "xccl_base_url: $xccl_base_url"

function check_files() {
    local url="$1"
    local local_dir="$2"
    echo "local dir: $local_dir"
    local local_file_name="${local_dir}.tar.gz"
    echo "local file name: $local_file_name"

    shift
    shift
    local files=("$@")

    # start to download
    echo "downloading: $url"
    rm -f ./$local_file_name
    wget -q $url -O ${local_file_name}
    if [[ $? -ne 0 ]]; then
        echo "downloading failed: $url"
        return 1
    else
        echo "downloading ok: $url"
    fi

    # remove local dir and de-compress
    rm -rf ./$local_dir
    tar xf $local_file_name
    if [[ $? -ne 0 ]]; then
        echo "de-compress failed: $local_file_name"
        return 1
    fi

    for i in "${files[@]}";
    do
        echo "checking $local_dir/$i"
        if [[ ! -f $local_dir/$i ]]; then
            echo "checking failed: $local_dir/$i"
            return 1
        else
            echo "checking ok: $local_dir/$i"
        fi
    done

    # clean
    rm -f ./$local_file_name
    rm -rf ./$local_dir
}

# XRE
xre_tar_file_names=("xre-kylin_aarch64" "xre-bdcentos_x86_64" "xre-ubuntu_x86_64" "xre-centos7_x86_64")
xre_inner_file_names=("include/xpu/runtime.h" "so/libxpurt.so")
for name in ${xre_tar_file_names[@]}; do
    url="${xpu_base_url}/${name}.tar.gz"
    check_files $url $name "${xre_inner_file_names[@]}"
    if [[ $? -ne 0 ]]; then
        echo "XRE check failed, name: $name"
        exit 1
    else
        echo "XRE check ok, name: $name"
    fi
done

# XDNN
xdnn_tar_file_names=("xdnn-kylin_aarch64" "xdnn-bdcentos_x86_64" "xdnn-ubuntu_x86_64" "xdnn-centos7_x86_64")
xdnn_inner_file_names=("include/xpu/xdnn.h" "so/libxpuapi.so")
for name in ${xdnn_tar_file_names[@]}; do
    url="${xpu_base_url}/${name}.tar.gz"
    check_files $url $name "${xdnn_inner_file_names[@]}"
    if [[ $? -ne 0 ]]; then
        echo "XDNN check failed, name: $name"
        exit 1
    else
        echo "XDNN check ok, name: $name"
    fi
done

# XCCL
xccl_tar_file_names=("xccl_rdma-bdcentos_x86_64" "xccl_rdma-ubuntu_x86_64" "xccl_socket-bdcentos_x86_64" "xccl_socket-kylin_aarch64" "xccl_socket-ubuntu_x86_64")
xccl_inner_file_names=("include/bkcl.h" "so/libbkcl.so")
for name in ${xccl_tar_file_names[@]}; do
    url="${xccl_base_url}/${name}.tar.gz"
    check_files $url $name "${xccl_inner_file_names[@]}"
    if [[ $? -ne 0 ]]; then
        echo "XCCL check failed, name: $name"
        exit 1
    else
        echo "XCCL check ok, name: $name"
    fi
done

echo "ALL CHECKS PASSED"

exit 0
