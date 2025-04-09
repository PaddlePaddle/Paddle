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

global_total_count=0
global_success_count=0
global_exit_250_arr=()
global_runtime_fail_arr=()
global_verification_fail_arr=()

install_paddle(){
    echo -e "\033[31m ---- Install paddlepaddle-gpu  \033"
    if [ -n "$paddle" ];then
      python -m pip install --user --no-cache-dir ${paddle} --force-reinstall --no-dependencies;
    fi
    python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";
}

install_external_ops(){
    echo -e "\033[31m ---- Install extern_ops  \033"
    export PYTHONPATH=/workspace/PaddleNLP:$PYTHONPATH
    cd /workspace/PaddleNLP/slm/model_zoo/gpt-3/external_ops
    python setup.py install
    python -c "import fused_ln;";
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

# There are two types of tests included here:
# 1. The auto-parallel unit testing in Paddle repository. CI will immediately end with
# an error when a test fails.
# 2. The auto-parallel testing of large language models in `PaddleNLP` repository. The execution
# status of each test will be recorded through global variables. When a test fails, it does not
# affect the execution of subsequent tests. They will be summarized and output after the CI is completed.
# Therefore, it is required to perform paddle unit testing first, followed by testing in `PaddleNLP`.
case_list[${#case_list[*]}]="llama_auto_unit_test"
case_list[${#case_list[*]}]=llama_auto
case_list[${#case_list[*]}]=gpt-3_auto
case_list[${#case_list[*]}]=gpt-3_dygraph
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
    echo -e "\033[32m The $3 CI has completed \033"
fi
}

function execute_func_list(){
    cd ${log_path} || { echo "Failed to enter log_path: $log_path"; return 1; }
    total_count=0
    success_count=0
    runtime_fail_count=0
    verification_fail_count=0
    exit_250_count=0
    while IFS= read -r func_name; do
        let total_count++
        let global_total_count++
        execute_num=1
        while true; do
            bash $1 exec_case $func_name $FLAGS_install_deps $FLAGS_download_data
            result=$?
            if [ $result -eq 0 ]; then
                echo -e "\033[32m test success!"
                let success_count++
                let global_success_count++
            elif [ $result -eq 2 ]; then
                echo -e "\033[31m verification failed!"
                let verification_fail_count++
                global_verification_fail_arr+=("$func_name")
            elif [ $result -eq 250 ]; then
                if [ $execute_num -eq 1 ]; then
                    echo -e "\033[31m fist time execute failed, try again!"
                    let execute_num++
                    continue
                else
                    echo -e "\033[31m second time execute failed, exit!"
                    let exit_250_count++
                    global_exit_250_arr+=("$func_name")
                fi
            else
                echo "test failed!"
                mv ${log_path}/$func_name ${log_path}/${func_name}_FAIL.log
                echo -e "\033[31m ${log_path}/$func_name_FAIL \033"
                tail -15 ${log_path}/${func_name}_FAIL.log
                let runtime_fail_count++
                global_runtime_fail_arr+=("$func_name")
            fi
            break
        done
    done < functions.txt
    echo -e "\033[31m $2 test case has complicated \033"
    echo -e "\033[31m $(printf '\t')  total tests :  $total_count \033"
    echo -e "\033[31m $(printf '\t')  success tests :  $success_count \033"
    echo -e "\033[31m $(printf '\t')  runtime fail tests :  $runtime_fail_count \033"
    echo -e "\033[31m $(printf '\t')  verification fail tests :  $verification_fail_count \033"
    echo -e "\033[31m $(printf '\t')  exit 250 tests(intermittent issue) :  $exit_250_count \033"
}

function clean_file(){
    target_path=$1
    matching_data_dirs=$(find "$target_path" -maxdepth 1 -type d -name "*data*")
    if [ -n "$matching_data_dirs" ]; then
        echo "cleaning data dirs:"
        echo $matching_data_dirs
        for dir in $matching_data_dirs; do
            rm -rf "$dir"
            echo "deleted $dir"
        done
    else
        echo "$target_path no data dirs found"
    fi

    matching_output_dirs=$(find "$target_path" -maxdepth 1 -type d -name "*output*")
    if [ -n "$matching_output_dirs" ]; then
        echo "cleaning output dirs:"
        echo $matching_output_dirs
        for dir in $matching_output_dirs; do
            rm -rf "$dir"
            echo "deleted $dir"
        done
    else
        echo "$target_path no output dirs found"
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
    # Install external_ops
    install_external_ops
    case_num=1
    # `FLAGS_install_deps` defaults to 0, indicating that certain required packages must be installed
    # via `install requirements.txt` prior to running `PaddleNLP` tests.
    # By setting `FLAGS_install_deps` to 1, indicating that there is no need for reinstallation.
    export FLAGS_install_deps=0
    for case in ${case_list[*]};do
        echo -e "\033[31m ---- running case $case_num/${#case_list[*]}: ${case} \033"
        # Suggest that the logical order here be consistent with the `case_stst` order.
        if [[ ${case} == "auto_unit_test" ]];then
            bash /workspace/Paddle/tools/auto_parallel/ci_case_unit.sh auto_unit_test
            print_info $? `ls -lt ${log_path} | grep "test" | head -n 1 | awk '{print $9}'` ${case}
            let case_num++
        elif [[ ${case} == "dygraph_unit_test" ]];then
            bash /workspace/Paddle/tools/auto_parallel/ci_case_unit.sh dygraph_unit_test
            print_info $? `ls -lt ${log_path} | grep "test" | head -n 1 | awk '{print $9}'` ${case}
            let case_num++
        elif [[ ${case} == "llama_auto_unit_test" ]];then
            bash /workspace/Paddle/tools/auto_parallel/ci_case_unit.sh llama_auto_unit_test
            print_info $? `ls -lt ${log_path} | grep "test" | head -n 1 | awk '{print $9}'` ${case}
            let case_num++
        elif [[ ${case} == "llama_auto" ]];then
            cmd=/workspace/PaddleNLP/scripts/distribute/ci_case_auto.sh
            bash $cmd prepare_case llama_case_list_auto $FLAGS_install_deps $FLAGS_download_data
            execute_func_list $cmd llama_auto
            # There is no need to reinstall the related packages of `PaddleNLP` afterward.
            export FLAGS_install_deps=1
            # The `llama` test data has been downloaded, and the `FLAGS_download_data` flag indicates
            # that there is no need to repeat the download process later.
            export FLAGS_download_data="llama ""$FLAGS_download_data"
            let case_num++
            clean_file /workspace/PaddleNLP/llm/auto_parallel/llama
        elif [[ ${case} == "gpt-3_auto" ]];then
            cmd=/workspace/PaddleNLP/scripts/distribute/ci_case_auto.sh
            bash $cmd prepare_case llm_gpt_case_list_auto $FLAGS_install_deps $FLAGS_download_data
            execute_func_list $cmd gpt-3_auto
            # there is no need to repeat the `gpt` download process later.
            export FLAGS_download_data="gpt ""$FLAGS_download_data"
            let case_num++
            clean_file /workspace/PaddleNLP/llm/auto_parallel/gpt-3
        elif [[ ${case} == "gpt-3_dygraph" ]];then
            cmd=/workspace/PaddleNLP/scripts/distribute/ci_case_dy.sh
            bash $cmd prepare_case llm_gpt_case_list_dygraph $FLAGS_install_deps $FLAGS_download_data
            execute_func_list $cmd gpt-3_dygraph
            let case_num++
            clean_file /workspace/PaddleNLP/llm
        else
            echo -e "\033[31m ---- no ${case} \033"
            let case_num++
        fi
    done
    echo -e "\033[31m ---- end run case  \033"

    echo -e "\033[31m ---- total tests :  $global_total_count \033"
    if [ ${#global_exit_250_arr[@]} -ne 0 ]; then
        echo -e "\033[32m ---- exit 250 test  :  ${#global_exit_250_arr[@]} \033"
        for case in "${global_exit_250_arr[@]}"; do
            echo -e "\t$case(exit 250)"
        done
    fi

    if [ ${#global_runtime_fail_arr[@]} -eq 0 ] && [ ${#global_verification_fail_arr[@]} -eq 0 ]; then
        echo -e "\033[32m ---- all cases Success  \033"
        EXCODE=0
    else
        echo -e "\033[32m ---- runtime failed test  :  ${#global_runtime_fail_arr[@]} \033"
        for case in "${global_runtime_fail_arr[@]}"; do
            echo -e "\t$case(failed)"
        done
        echo -e "\033[32m ---- verification failed test  :  ${#global_verification_fail_arr[@]} \033"
        for case in "${global_verification_fail_arr[@]}"; do
            echo -e "\t$case(failed)"
        done
        EXCODE=1
    fi
else
    echo -e "\033[32m Changed Not CI case, Skips \033"
    EXCODE=0
fi
exit $EXCODE
