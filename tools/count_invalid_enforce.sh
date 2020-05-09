#!/bin/bash

# This script is used to count all PADDLE checks in the paddle/fluid directory,
#   including the total PADDLE check number, the valid check number and the
#   invalid check number under paddle/fluid and its main sub-directories.

# Usage: bash count_invalid_enforce.sh (run in tools directory)

# Result Example:

#     paddle/fluid/framework - total: 1065, valid: 267, invalid: 798
#     paddle/fluid/imperative - total: 135, valid: 118, invalid: 17
#     paddle/fluid/inference - total: 449, valid: 158, invalid: 291
#     paddle/fluid/memory - total: 60, valid: 10, invalid: 50
#     paddle/fluid/operators - total: 4225, valid: 1061, invalid: 3164
#     paddle/fluid/platform - total: 240, valid: 39, invalid: 201
#     paddle/fluid/pybind - total: 98, valid: 53, invalid: 45
#     paddle/fluid/string - total: 0, valid: 0, invalid: 0
#     paddle/fluid/testdata - total: 0, valid: 0, invalid: 0
#     paddle/fluid/train - total: 6, valid: 0, invalid: 6
#     ----------------------------
#     PADDLE ENFORCE & THROW COUNT
#     ----------------------------
#     All PADDLE_ENFORCE{_**} & PADDLE_THROW Count: 6278
#     Valid PADDLE_ENFORCE{_**} & PADDLE_THROW Count: 1706
#     Invalid PADDLE_ENFORCE{_**} & PADDLE_THROW Count: 4572

ROOT_DIR=../paddle/fluid
ALL_PADDLE_CHECK_CNT=0
VALID_PADDLE_CHECK_CNT=0

function enforce_scan(){
    paddle_check=`grep -r -zoE "(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\(.[^,\);]*.[^;]*\);\s" $1 || true`
    total_check_cnt=`echo "$paddle_check" | grep -cE "(PADDLE_ENFORCE|PADDLE_THROW)" || true`
    valid_check_cnt=`echo "$paddle_check" | grep -zoE '(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\((.[^,;]+,)*.[^";]*(errors::).[^"]*".[^";]{20,}.[^;]*\);\s' | grep -cE "(PADDLE_ENFORCE|PADDLE_THROW)" || true`
    eval $2=$total_check_cnt
    eval $3=$valid_check_cnt
}

function walk_dir(){
    for file in `ls $1`
    do
        if [ -d $1"/"$file ];then
            level=$(($2+1))
            if [ $level -le 1 ]; then
                enforce_scan $1"/"$file total_check_cnt valid_check_cnt
                dir_name=$1
                echo "${dir_name#../}"/"$file - total: ${total_check_cnt}, valid: ${valid_check_cnt}, invalid: $(($total_check_cnt-$valid_check_cnt))"
                ALL_PADDLE_CHECK_CNT=$(($ALL_PADDLE_CHECK_CNT+$total_check_cnt))
                VALID_PADDLE_CHECK_CNT=$(($VALID_PADDLE_CHECK_CNT+$valid_check_cnt))
                walk_dir $1"/"$file $level
            fi
        fi
    done
}

walk_dir $ROOT_DIR 0

echo "----------------------------"
echo "PADDLE ENFORCE & THROW COUNT"
echo "----------------------------"
echo "All PADDLE_ENFORCE{_**} & PADDLE_THROW Count: ${ALL_PADDLE_CHECK_CNT}"
echo "Valid PADDLE_ENFORCE{_**} & PADDLE_THROW Count: ${VALID_PADDLE_CHECK_CNT}"
echo "Invalid PADDLE_ENFORCE{_**} & PADDLE_THROW Count: $(($ALL_PADDLE_CHECK_CNT-$VALID_PADDLE_CHECK_CNT))"
