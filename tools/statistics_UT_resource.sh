# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

unset GREP_OPTIONS
rm ./run_detail.log
rm ./UT_resource.log
rm ./UT_resource_sort.log
rm ./while_list.log

export LD_LIBRARY_PATH="$PWD/python/paddle/libs;$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0,1

test_cases=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d')
use_memory_base=$(nvidia-smi -q -i 0  | grep "Used"  | head -1 | grep -o "[0-9]*")
for unittest in $test_cases
do
    use_memory=0
    gpu_utilization=0
    memory_utilization=0
    ctest -R "^${unittest}$" --repeat-until-fail 5 -j 1 &
    PID=$!
    echo -e "******************************************************"
    echo -e "[$unittest]:    PID:$PID \n"
    while [[ $(ps aux | awk '{print $2}' | grep "^$PID$" | grep -v "grep" | wc -l) -ge 1 ]]
    do
        use_memory_current=$(nvidia-smi -q -i 0  | grep "Used"  | head -1 | grep -o "[0-9]*")
        if [[ $use_memory_current -gt $use_memory ]];then
            use_memory=$use_memory_current
        fi
        memory_utilization_current=$(nvidia-smi -q -i 0 |  grep "Memory" | sed -n '3p' | grep -o "[0-9]*")
        if [[ $memory_utilization_current -gt $memory_utilization ]];then
            memory_utilization=$memory_utilization_current
        fi

        gpu_utilization_current=$(nvidia-smi -q -i 0  | grep "Gpu"  | grep -o "[0-9]*")
        if [[ $gpu_utilization_current -gt $gpu_utilization ]];then
            gpu_utilization=$gpu_utilization_current
        fi
    done
    use_memory=`expr $use_memory - $use_memory_base`
    echo -e "     use_memory:$use_memory \n     memory_utilization:$memory_utilization \n     gpu_utilization:$gpu_utilization\n"
    echo -e "[$unittest]: \n     use_memory:$use_memory \n     memory_utilization:$memory_utilization \n     gpu_utilization:$gpu_utilization\n" >> run_detail.log
    echo -e "$unittest:$use_memory:$memory_utilization:$gpu_utilization" >> UT_resource.log
done

sort -r -n -k 2 -t : UT_resource.log > UT_resource_sort.log
cat UT_resource_sort.log | awk -F ':' '{print $1}' > while_list.log
