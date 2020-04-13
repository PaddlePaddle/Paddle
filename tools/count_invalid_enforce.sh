#!/bin/bash

ROOT_DIR=../paddle/fluid
ALL_PADDLE_CHECK_CNT=0
VALID_PADDLE_CHECK_CNT=0

function enforce_scan(){
    paddle_check=`grep -r -zoE "(PADDLE_ENFORCE[A-Z_]*|PADDLE_THROW)\(.[^,\);]*.[^;]*\);\s" $1 || true`
    total_check_cnt=`echo "$paddle_check" | grep -cE "(PADDLE_ENFORCE|PADDLE_THROW)" || true`
    valid_check_cnt=`echo "$paddle_check" | grep -zoE '(PADDLE_ENFORCE[A-Z_]*|PADDLE_THROW)\((.[^,;]+,)*.[^";]*(errors::).[^"]*".[^";]{20,}.[^;]*\);\s' | grep -cE "(PADDLE_ENFORCE|PADDLE_THROW)" || true`
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
                echo $1"/"$file" - total: ${total_check_cnt}, valid: ${valid_check_cnt}, invalid: $(($total_check_cnt-$valid_check_cnt))"
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
