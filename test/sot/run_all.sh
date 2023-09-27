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

# 遍历目录下的所有 python 文件
export PYTHONPATH=$PYTHONPATH:../
export STRICT_MODE=1
export COST_MODEL=False
export MIN_GRAPH_SIZE=0

failed_tests=()

for file in ./test_*.py; do
    # 检查文件是否为 python 文件
    if [ -f "$file" ]; then
        if [[ -n "$GITHUB_ACTIONS" ]]; then
            echo ::group::Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=1 python " $file
        else
            echo Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=1 python " $file
        fi
        # 执行文件
        python_output=$(python $file 2>&1)

        if [ $? -ne 0 ]; then
            echo "run $file failed"
            failed_tests+=("$file")
            if [[ -n "$GITHUB_ACTIONS" ]]; then
                echo -e "$python_output" | python ./extract_errors.py
            fi
            echo -e "$python_output"
        fi
        if [[ -n "$GITHUB_ACTIONS" ]]; then
            echo "::endgroup::"
        fi
    fi
done

if [ ${#failed_tests[@]} -ne 0 ]; then
    echo "failed tests file:"
    for failed_test in "${failed_tests[@]}"; do
        echo "$failed_test"
    done
    exit 1
fi
