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

# Insatll paddlepaddle-gpu
install_paddle(){
    echo -e "\033[31m ---- Install paddlepaddle-gpu  \033"
    python -m pip install --user ${paddle} --force-reinstall --no-dependencies;
    python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";
}

get_diff_TO_case(){
cd ${paddle_dir}
for file_name in `git diff --numstat upstream/${AGILE_COMPILE_BRANCH} |awk '{print $NF}'`;do
    arr_file_name=(${file_name//// })
    dir1=${arr_file_name[0]}
    dir2=${arr_file_name[1]}
    dir3=${arr_file_name[2]}
    dir4=${arr_file_name[3]}
    file_item=$dir1/$dir2/$dir3/$dir4
    echo "file_name:"${file_name}, "path:"${file_item}
    if [ ! -f ${file_name} ];then # 针对pr删掉文件
        continue
    elif [[ ${file_name##*.} == "md" ]] || [[ ${file_name##*.} == "rst" ]] || [[ ${dir1} == "docs" ]];then
        continue
    else
        for ((i=0; i<${#target_lists_for_hybrid_ci[@]}; i++)); do
            if [[ ${file_item} == *${target_lists_for_hybrid_ci[i]}* ]];then
                case_list[${#case_list[*]}]=gpt-3
                case_list[${#case_list[*]}]=unit_test
                break
            else
                continue
            fi
        done
    fi

done
}

print_info(){
if [ $1 -ne 0 ];then
    EXCODE=2
    if [ ! -f ${log_path}/$2 ];then
        echo -e "\033[31m run CI FAIL \033"
    else
        mv ${log_path}/$2 ${log_path}/$2_FAIL.log
        echo -e "\033[31m ${log_path}/$2_FAIL \033"
        tail -10 ${log_path}/$2_FAIL.log
    fi
    exit $EXCODE
else
    echo -e "\033[32m run CI SUCCESS \033"
fi
}

get_diff_TO_case # 获取待执行case列表
case_list=($(awk -v RS=' ' '!a[$1]++' <<< ${case_list[*]}))  # 去重并将结果存储回原列表
if [[ ${#case_list[*]} -ne 0 ]];then
    echo -e "\033[31m =======CI Check case========= \033"
    echo -e "\033[31m ---- case_list length: ${#case_list[*]}, cases: ${case_list[*]} \033"
    set +e
    echo -e "\033[31m ---- start run case  \033"
    
    # Install paddle
    install_paddle
    case_num=1
    for case in ${case_list[*]};do
        echo -e "\033[31m ---- running case $case_num/${#case_list[*]}: ${case} \033"
        if [[ ${case} == "gpt-3" ]];then
            echo -e "\033[31m ---- running case gpt-3 auto \033"
            bash /workspace/PaddleNLP/scripts/distribute/ci_case_auto.sh
            print_info $? `ls -lt ${log_path} | grep gpt | head -n 1 | awk '{print $9}'`
            let case_num++
        elif [[ ${case} == "unit_test" ]];then
            echo -e "\033[31m ---- running case unit_test \033"
            bash /workspace/Paddle/tools/auto_parallel/ci_case_unit.sh
            print_info $? `ls -lt ${log_path} | grep test | head -n 1 | awk '{print $9}'`
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
        echo -e "\033[32m ---- case Success \033"
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
