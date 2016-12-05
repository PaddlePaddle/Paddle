#!/bin/bash

set -e
cd `dirname $0`
export PYTHONPATH=$PWD/../../../../

protostr=$PWD/protostr
. file_list.sh

for conf in ${configs[*]}
do
    echo "Generating " $conf
    python -m paddle.utils.dump_config $conf.py > $protostr/$conf.protostr.unittest
done

for conf in ${whole_configs[*]}
do
    echo "Generating " $conf
    python -m paddle.utils.dump_config $conf.py "" --whole > $protostr/$conf.protostr.unittest
done
