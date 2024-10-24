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

function check_xre() {
    local xre_version="$1"
    local xre_base_url="https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release"
    local xre_tar_file_names=("xre-ubuntu_2004-x86_64" "xre-bdcentos-x86_64")
    local xre_files=("include/xpu/runtime.h" "so/libxpurt.so")
    for name in ${xre_tar_file_names[@]}; do
        local xre_url="${xre_base_url}/${xre_version}/${name}-${xre_version}.tar.gz"
        check_files "${xre_url}" "${name}-${xre_version}" "${xre_files[@]}"
        if [[ $? -ne 0 ]]; then
            echo "XRE check failed, name: $name"
            exit 1
        else
            echo "XRE check ok, name: $name"
        fi
    done
}

function check_xhpc() {
    local xhpc_date="$1"
    local xhpc_base_url="https://klx-sdk-release-public.su.bcebos.com/xhpc/dev"
    local xhpc_tar_file_names=("xhpc-ubuntu2004_x86_64" "xhpc-bdcentos7_x86_64")
    local xhpc_files=("xblas/include/cublasLt.h" "xblas/so/libxpu_blas.so" "xdnn/include/xpu/xdnn.h" "xdnn/so/libxpuapi.so" "xfa/include/flash_api.h" "xfa/so/libxpu_flash_attention.so" "xpudnn/include/xpudnn.h" "xpudnn/so/libxpu_dnn.so")
    for name in ${xhpc_tar_file_names[@]}; do
        local xhpc_url="${xhpc_base_url}/${xhpc_date}/${name}.tar.gz"
        check_files "${xhpc_url}" "${name}" "${xhpc_files[@]}"
        if [[ $? -ne 0 ]]; then
            echo "XHPC check failed, name: $name"
            exit 1
        else
            echo "XHPC check ok, name: $name"
        fi
    done
}

function check_xccl() {
    local xccl_version="$1"
    local xccl_base_url="https://klx-sdk-release-public.su.bcebos.com/xccl/release"
    local xccl_tar_file_names=("xccl_rdma-bdcentos_x86_64" "xccl_rdma-ubuntu_x86_64" "xccl_socket-bdcentos_x86_64" "xccl_socket-ubuntu_x86_64")
    local xccl_files=("include/bkcl.h" "so/libbkcl.so")
    for name in ${xccl_tar_file_names[@]}; do
        local xccl_url="${xccl_base_url}/${xccl_version}/${name}.tar.gz"
        check_files "${xccl_url}" "${name}" "${xccl_files[@]}"
        if [[ $? -ne 0 ]]; then
            echo "XCCL check failed, name: $name"
            exit 1
        else
            echo "XCCL check ok, name: $name"
        fi
    done
}

ARGS=$(getopt -o:: --long xre:,xhpc:,xccl:: -n "check_xpu_dependence" -- "$@")
eval set -- "${ARGS}"

while true; do
    case "$1" in
        --xre)
            check_xre $2; shift 2 ;;
        --xhpc)
            check_xhpc $2; shift 2 ;;
        --xccl)
            check_xccl $2; shift 2 ;;
        --)
            shift; break ;;
        *)
            echo "unrecognized option: $1"
            exit 1 ;;
    esac

done

echo "ALL CHECKS PASSED"

exit 0
