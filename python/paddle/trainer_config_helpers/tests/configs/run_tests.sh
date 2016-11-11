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
    diff $base_protostr $new_protostr -u
done
