#!/bin/bash

# This script is used to count PADDLE checks by files in the paddle/fluid/operators directory,
#   contains the number of PADDLE checks under each file.
#   
#   The three columns of data are: total number, valid number, invalid number. 
#   The output format is easy to display as a markdown table.

# Usage: bash file_invalid_enforce.sh (run in tools directory)

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

ROOT_DIR=../paddle/fluid/operators

function enforce_scan(){
    paddle_check=`grep -r -zoE "(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\(.[^,\);]*.[^;]*\);\s" $1 || true`
    total_check_cnt=`echo "$paddle_check" | grep -cE "(PADDLE_ENFORCE|PADDLE_THROW)" || true`
    valid_check_cnt=`echo "$paddle_check" | grep -zoE '(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\((.[^,;]+,)*.[^";]*(errors::).[^"]*".[^";]{20,}.[^;]*\);\s' | grep -cE "(PADDLE_ENFORCE|PADDLE_THROW)" || true`
    eval $2=$total_check_cnt
    eval $3=$valid_check_cnt
}

function walk_dir(){
    dir_name=$1
    echo "**${dir_name#../}** | **$2** | **$3** | **$(($2-$3))**"
    local i=0
    local dir_array
    for file in `ls $1`
    do
        if [ -f $1"/"$file ];then
            enforce_scan $1"/"$file file_total_check_cnt file_valid_check_cnt
            file_invalid_check_cnt=$(($total_check_cnt-$valid_check_cnt))
            if [ $file_invalid_check_cnt -gt 0 ];then
                echo "- $file | ${file_total_check_cnt} | ${file_valid_check_cnt} | ${file_invalid_check_cnt}"
            fi
        fi
        if [ -d $1"/"$file ];then
            
            dir_array[$i]=$1"/"$file
            ((i++))
        fi
    done
    for sub_dir_name in ${dir_array[@]}
    do
        enforce_scan $sub_dir_name dir_total_check_cnt dir_valid_check_cnt
        walk_dir $sub_dir_name $dir_total_check_cnt $dir_valid_check_cnt
    done
}

walk_dir $ROOT_DIR 0 0
