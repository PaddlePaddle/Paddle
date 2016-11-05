#!/bin/bash
cd `dirname $0`
set -e

protostr=`dirname $0`/protostr
files=`ls $protostr | grep -v "unitest"`

./generate_protostr.sh

for file in $files
do
    base_protostr=$protostr/$file
    new_protostr=$protostr/$file.unitest
    if [ -f $new_protostr ];then
        base_md5=`md5sum $base_protostr | awk '{print $1}'`
        new_md5=`md5sum $new_protostr | awk '{print $1}'`
        if [ $base_md5 != $new_md5 ];then
            echo $new_protostr: FAILED!
        else
            echo $new_protostr: OK!
        fi
    else
        echo WARNING $new_protostr: NOT EXIST!
    fi
done
