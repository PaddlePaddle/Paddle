#!/bin/bash

set -e
cd `dirname $0`

protostr=$PWD/protostr
. file_list.sh

for conf in ${configs[*]}
do
    echo "Generating " $conf
    $1 -m paddle.utils.dump_config $conf.py > $protostr/$conf.protostr.unittest
    if [ ! -f "$protostr/$conf.protostr" ]; then 
        cp $protostr/$conf.protostr.unittest $protostr/$conf.protostr
    fi
    cat ${conf}.py |$1 test_config_parser_for_non_file_config.py > $protostr/$conf.protostr.non_file_config.unittest
done

for conf in ${whole_configs[*]}
do
    echo "Generating " $conf
    $1 -m paddle.utils.dump_config $conf.py "" --whole > $protostr/$conf.protostr.unittest
    if [ ! -f "$protostr/$conf.protostr" ]; then 
        cp $protostr/$conf.protostr.unittest $protostr/$conf.protostr
    fi
    cat ${conf}.py |$1 test_config_parser_for_non_file_config.py --whole > $protostr/$conf.protostr.non_file_config.unittest
done
