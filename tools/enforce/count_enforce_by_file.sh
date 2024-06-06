#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# This script is used to count PADDLE checks by files in the paddle/fluid/operators directory,
#   contains the number of PADDLE checks under each file.
#
#   The three columns of data are: total number, valid number, invalid number.
#   The output format is easy to display as a markdown table.

# Usage: bash count_enforce_by_file.sh  [target directory or file] (run in tools directory)
#   - The default check path is paddle/fluid/operators

# Result Example:

#   **paddle/fluid/operators/math** | **200** | **7** | **193**
#   - beam_search.cc | 1 | 0 | 1
#   - beam_search.cu | 1 | 0 | 1
#   - blas.cc | 1 | 0 | 1
#   - blas_impl.cu.h | 8 | 1 | 7
#   - blas_impl.h | 15 | 0 | 15
#   - concat_test.cc | 16 | 0 | 16
#   - context_project.h | 1 | 0 | 1
#   - cpu_vec.h | 1 | 0 | 1
#   - cross_entropy.cu | 1 | 0 | 1
#   - cross_entropy.h | 1 | 0 | 1
#   - im2col.cc | 12 | 0 | 12
#   - im2col.cu | 12 | 0 | 12
#   - math_function.cc | 2 | 0 | 2
#   - math_function.cu | 4 | 0 | 4
#   - math_function_impl.h | 10 | 0 | 10

. ./count_all_enforce.sh --source-only

ROOT_DIR=../paddle/fluid/operators

if [ "$1" != "" ]; then
    ROOT_DIR=$1
fi

FILE_WHITE_LIST="\
    box_clip_op.cc \
    box_clip_op.h \
    elementwise_op_function.cu.h \
    fused_elemwise_activation_op.cc \
    auc_op.cu \
    unsqueeze_op.h \
    unsqueeze_op.cc \
    enforce.h \
    errors_test.cc \
    cross_entropy.cu \
    cross_entropy.h \
    unpooling.cu"

function count_file_recursively(){
    dir_name=$1
    echo "**${dir_name#../}** | **$2** | **$3** | **$(($2-$3))**"
    local i=0
    local dir_array
    for file in `ls $1`
    do
        if [ -f $1"/"$file ];then
            in_white_list=$(echo $FILE_WHITE_LIST | grep "${file}")
            if [[ "$in_white_list" == "" ]];then
                enforce_count $1"/"$file file_total_check_cnt file_valid_check_cnt
                file_invalid_check_cnt=$(($total_check_cnt-$valid_check_cnt))
                if [ $file_invalid_check_cnt -gt 0 ];then
                    echo "- $file | ${file_total_check_cnt} | ${file_valid_check_cnt} | ${file_invalid_check_cnt}"
                fi
            fi
        fi
        if [ -d $1"/"$file ];then
            dir_array[$i]=$1"/"$file
            ((i++))
        fi
    done
    for sub_dir_name in ${dir_array[@]}
    do
        enforce_count $sub_dir_name dir_total_check_cnt dir_valid_check_cnt
        count_file_recursively $sub_dir_name $dir_total_check_cnt $dir_valid_check_cnt
    done
}

main() {
    count_file_recursively $ROOT_DIR 0 0
}

if [ "${1}" != "--source-only" ]; then
    main "${@}"
fi
