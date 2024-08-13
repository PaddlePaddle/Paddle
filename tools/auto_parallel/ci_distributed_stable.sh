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

source /workspace/Paddle/tools/auto_parallel/target_path_lists.sh

export paddle=$1
export paddle_dir=/workspace/Paddle
mkdir -p /workspace/case_logs
export log_path=/workspace/case_logs
export case_list=()

install_paddle(){
    echo -e "\033[31m ---- Install paddlepaddle-gpu  \033"
    if [ -n "$paddle" ];then
      python -m pip install --user ${paddle} --no-dependencies;
    fi
    python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";
}

get_diff_TO_case(){
cd ${paddle_dir}
# get the location of "test/auto_parallel" in target_lists_for_semi_auto_ci
count=0
for element in "${target_lists_for_semi_auto_ci[@]}";do
  if [[ "$element" == "test/auto_parallel" ]]; then
    test_auto_num=$count
    break
  fi
  count=$((count+1))
done
# get the location of "test/collective/hybrid_strategy" in target_lists_for_dygraph_ci
count=0
for element in "${target_lists_for_dygraph_ci[@]}";do
  if [[ "$element" == "test/collective/hybrid_strategy" ]]; then
    test_dygraph_num=$count
    break
  fi
  count=$((count+1))
done
for file_name in `git diff --numstat upstream/${AGILE_COMPILE_BRANCH} |awk '{print $NF}'`;do
    arr_file_name=(${file_name//// })
    dir1=${arr_file_name[0]}
    dir2=${arr_file_name[1]}
    dir3=${arr_file_name[2]}
    dir4=${arr_file_name[3]}
    dir5=${arr_file_name[4]}
    dir6=${arr_file_name[5]}
    file_item=$dir1/$dir2/$dir3/$dir4/$dir5/$dir6
    echo "file_name:"${file_name}, "path:"${file_item}
    if [ ! -f ${file_name} ];then # deleting files for PR
        continue
    elif [[ ${file_name##*.} == "md" ]] || [[ ${file_name##*.} == "rst" ]] || [[ ${dir1} == "docs" ]];then
        continue
    else
        # The most auto unittests have been monitored in PR-CI-Distribute-stable,
        # while the other tests of llama model will be executed in PR-CI-Auto-Parallel.
        for ((i=0; i<${#target_lists_for_semi_auto_ci[@]}; i++)); do
            if [[ ${file_item} == *${target_lists_for_semi_auto_ci[i]}* ]];then
                case_list[${#case_list[*]}]="auto_unit_test"
                break
            else
                continue
            fi
        done
        # The dynamic unittests have been monitored in PR-CI-Distribute-stable
        # and will be no longer redundantly executed in PR-CI-Auto-Parallel.
        for ((i=0; i<${#target_lists_for_dygraph_ci[@]}; i++)); do
            if [[ ${file_item} == *${target_lists_for_dygraph_ci[i]}* ]];then
                case_list[${#case_list[*]}]="dygraph_unit_test"
                break
            else
                continue
            fi
        done
    fi
done
}

print_info(){
#解决异常退出-6的问题，CI中的偶现问题，无法发现
if [[ $1 -ne 0 ]] && [[ $1 -ne 250 ]];then
    EXCODE=2
    if [ ! -f ${log_path}/$2 ];then
        echo -e "\033[31m run $2 CI FAIL \033"
    else
        mv ${log_path}/$2 ${log_path}/$2_FAIL.log
        echo -e "\033[31m ${log_path}/$2_FAIL \033"
        tail -70 ${log_path}/$2_FAIL.log
    fi
    exit $EXCODE
else
    echo -e "\033[32m run $3 CI SUCCESS \033"
fi
}

# Get the list of pending cases
get_diff_TO_case
# Remove duplicates and store the results back to the original list

####################
case_list=($(awk -v RS=' ' '!a[$1]++' <<< ${case_list[*]}))
if [[ ${#case_list[*]} -ne 0 ]];then
    echo -e "\033[31m =======CI Check case========= \033"
    echo -e "\033[31m ---- case_list length: ${#case_list[*]}, cases: ${case_list[*]} \033"
    echo -e "\033[31m ============================= \033"
    set +e

    # Install paddle
    install_paddle
    case_num=1
    export FLAGS_install_deps=0
    for case in ${case_list[*]};do
        echo -e "\033[31m ---- running case $case_num/${#case_list[*]}: ${case} \033"
        if [[ ${case} == "llama_auto" ]];then
            bash /workspace/PaddleNLP/scripts/distribute/ci_case_auto.sh llama_case_list_auto $FLAGS_install_deps $FLAGS_download_data
            print_info $? `ls -lt ${log_path} | grep "llama" | head -n 1 | awk '{print $9}'` ${case}
            export FLAGS_install_deps=1
            export FLAGS_download_data="llama ""$FLAGS_download_data"
            let case_num++
        elif [[ ${case} == "auto_unit_test" ]];then
            bash /workspace/Paddle/tools/auto_parallel/ci_case_unit.sh auto_unit_test
            print_info $? `ls -lt ${log_path} | grep "test" | head -n 1 | awk '{print $9}'` ${case}
            let case_num++
        elif [[ ${case} == "gpt-3_dygraph" ]];then
            bash /workspace/PaddleNLP/scripts/distribute/ci_case_dy.sh llm_gpt_case_list_dygraph $FLAGS_install_deps $FLAGS_download_data
            print_info $? `ls -lt ${log_path} | grep "llm_gpt" | head -n 1 | awk '{print $9}'` ${case}
            let case_num++
        elif [[ ${case} == "dygraph_unit_test" ]];then
            bash /workspace/Paddle/tools/auto_parallel/ci_case_unit.sh dygraph_unit_test
            print_info $? `ls -lt ${log_path} | grep "test" | head -n 1 | awk '{print $9}'` ${case}
            let case_num++
        elif [[ ${case} == "llama_auto_unit_test" ]];then
            bash /workspace/Paddle/tools/auto_parallel/ci_case_unit.sh llama_auto_unit_test
            print_info $? `ls -lt ${log_path} | grep "test" | head -n 1 | awk '{print $9}'` ${case}
            let case_num++
        else
            echo -e "\033[31m ---- no ${case} \033"
            let case_num++
        fi
    done
    echo -e "\033[31m ---- end run case  \033"
    cd ${log_path}
    if [ ! -f *FAIL* ];then
        FF=0
        EXCODE=0
        echo -e "\033[32m ---- all case Success \033"
    else
        FF=`ls *FAIL*|wc -l`
        EXCODE=2
        echo -e "\033[31m ---- case Failed number: ${FF} \033"
        ls *_FAIL*
    fi
else
    echo -e "\033[32m Changed Not CI case, Skips \033"
    EXCODE=0
fi
exit $EXCODE
