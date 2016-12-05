#!/bin/bash
cd `dirname $0`

set -e

protostr=`dirname $0`/protostr

files=`ls $protostr | grep -v "unittest"`

./generate_protostr.sh

. ./file_list.sh

if [ -z $1 ]; then
  for file in $files
  do
      base_protostr=$protostr/$file
      new_protostr=$protostr/$file.unittest
      diff $base_protostr $new_protostr -u
  done
else
  for file in ${configs[*]}
  do
    if ! $1 $protostr/$file.protostr $protostr/$file.protostr.unittest; then
      diff $protostr/$file.protostr $protostr/$file.protostr.unittest -u
    fi
  done

  for file in ${whole_configs[*]}
  do
    if ! $1 $protostr/$file.protostr $protostr/$file.protostr.unittest --whole; then
      diff $protostr/$file.protostr $protostr/$file.protostr.unittest -u
    fi
  done
fi
