#!/bin/bash
cd `dirname $0`

protostr=`dirname $0`/protostr

files=`ls $protostr | grep -v "unitest"`

./generate_protostr.sh

for file in $files
do
    base_protostr=$protostr/$file
    new_protostr=$protostr/$file.unitest
    if [ -f $new_protostr ];then
        diff $base_protostr $new_protostr -u
        if [ $? -eq 0 ];then
            echo $new_protostr: OK!
        fi
    else
        echo WARNING $new_protostr: NOT EXIST!
    fi
done
