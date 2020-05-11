#!/bin/bash

# This script is used to count detail PADDLE checks in the paddle/fluid directory,
#   contains the number of PADDLE checks under each folder, the statistical data 
#   does not include subdirectories, only covers all files under the current directory.
#   
#   The three columns of data are: total number, valid number, invalid number. 
#   The output format is easy to display as a markdown table.

# Usage: bash detail_invalid_enforce.sh (run in tools directory)

# Result Example:

#     paddle/fluid/operators/benchmark | 11 | 0 | 11
#     paddle/fluid/operators/collective | 28 | 1 | 27
#     paddle/fluid/operators/controlflow | 60 | 59 | 1
#     paddle/fluid/operators/detail | 2 | 0 | 2
#     paddle/fluid/operators/detection | 276 | 146 | 130
#     paddle/fluid/operators/distributed/brpc | 17 | 0 | 17
#     paddle/fluid/operators/distributed/grpc | 13 | 0 | 13
#     paddle/fluid/operators/distributed | 63 | 10 | 53
#     paddle/fluid/operators/distributed_ops | 88 | 6 | 82
#     paddle/fluid/operators/elementwise/mkldnn | 5 | 5 | 0
#     paddle/fluid/operators/elementwise | 29 | 20 | 9
#     paddle/fluid/operators/fused | 227 | 182 | 45
#     paddle/fluid/operators/jit/gen | 17 | 0 | 17
#     paddle/fluid/operators/jit/more/intrinsic | 0 | 0 | 0
#     paddle/fluid/operators/jit/more/mix | 1 | 0 | 1
#     paddle/fluid/operators/jit/more/mkl | 9 | 0 | 9
#     paddle/fluid/operators/jit/more | 0 | 0 | 0
#     paddle/fluid/operators/jit/refer | 8 | 0 | 8
#     paddle/fluid/operators/jit | 18 | 0 | 18
#     paddle/fluid/operators/lite | 2 | 2 | 0
#     paddle/fluid/operators/math/detail | 0 | 0 | 0
#     paddle/fluid/operators/math | 200 | 7 | 193
#     paddle/fluid/operators/metrics | 38 | 29 | 9
#     paddle/fluid/operators/mkldnn | 107 | 14 | 93
#     paddle/fluid/operators/nccl | 27 | 0 | 27
#     paddle/fluid/operators/optimizers | 214 | 50 | 164
#     paddle/fluid/operators/reader | 40 | 14 | 26
#     paddle/fluid/operators/reduce_ops | 8 | 8 | 0
#     paddle/fluid/operators/sequence_ops | 167 | 47 | 120
#     paddle/fluid/operators/tensorrt | 7 | 4 | 3
#     paddle/fluid/operators | 2144 | 999 | 1145

ROOT_DIR=../paddle/fluid

function enforce_scan(){
    paddle_check=`grep -r -zoE "(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\(.[^,\);]*.[^;]*\);\s" $1 || true`
    total_check_cnt=`echo "$paddle_check" | grep -cE "(PADDLE_ENFORCE|PADDLE_THROW)" || true`
    valid_check_cnt=`echo "$paddle_check" | grep -zoE '(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\((.[^,;]+,)*.[^";]*(errors::).[^"]*".[^";]{20,}.[^;]*\);\s' | grep -cE "(PADDLE_ENFORCE|PADDLE_THROW)" || true`
    eval $2=$total_check_cnt
    eval $3=$valid_check_cnt
}

function walk_dir(){
    local sub_dir_total_check_cnt=0
    local sub_dir_valid_check_cnt=0
    for file in `ls $1`
    do
        if [ -d $1"/"$file ];then
            enforce_scan $1"/"$file dir_total_check_cnt dir_valid_check_cnt
            sub_dir_total_check_cnt=$(($sub_dir_total_check_cnt+$dir_total_check_cnt))
            sub_dir_valid_check_cnt=$(($sub_dir_valid_check_cnt+$dir_valid_check_cnt))
            
            walk_dir $1"/"$file $dir_total_check_cnt $dir_valid_check_cnt 
        fi
    done
    total_check_cnt=$(($2-$sub_dir_total_check_cnt))
    valid_check_cnt=$(($3-$sub_dir_valid_check_cnt))
    dir_name=$1
    echo "${dir_name#../} | ${total_check_cnt} | ${valid_check_cnt} | $(($total_check_cnt-$valid_check_cnt))"
}

walk_dir $ROOT_DIR 0 0
