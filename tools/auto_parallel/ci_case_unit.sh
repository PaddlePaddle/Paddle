#!/usr/bin/env bash
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

export log_path=/workspace/case_logs
export auto_case_path=/workspace/Paddle/test/auto_parallel/hybrid_strategy
export dygraph_case_path=/workspace/Paddle/test/collective/hybrid_strategy

function case_list_unit() {
    if [ ! -f "testslist.csv" ]; then
        echo "文件 testslist.csv 不存在"
        exit -1
    fi
    
    for ((i=2; i<=`awk -F, 'END {print NR}' testslist.csv`; i++)); do
        item=`awk -F, 'NR=='$i' {print}' testslist.csv`
        case_name=`awk -F, 'NR=='$i' {print $1}' testslist.csv`
        echo "=========== $case_name run  begin ==========="
        if [[ $item =~ PYTHONPATH=([^,;]*)([,;]|$) ]]; then
            substring="${BASH_REMATCH[1]}"
            echo "PYTHONPATH=$substring"
            export PYTHONPATH=$substring:$PYTHNPATH
        fi
        python $case_name.py >>${log_path}/$case_name 2>&1
        if [ $? -eq 0 ]; then
            tail -n 10 ${log_path}/$case_name
        fi
        echo "=========== $case_name run  end ==========="
    done
}

main() {
    export exec_case=$1
    echo -e "\033[31m ---- Start executing $exec_case case \033[0m"

    if [[ $exec_case =~ "auto_unit_test" ]];then
        cd ${auto_case_path}
        case_list_unit
    elif [[ $exec_case =~ "dygraph_unit_test" ]];then
        cd ${dygraph_case_path}
        case_list_unit
    else
        echo -e "\033[31m ---- Invalid exec_case $exec_case \033[0m"
    fi
}

main $@
