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

export STRICT_MODE=0
export ENABLE_SOT=True
export ENABLE_FALL_BACK=True
export COST_MODEL=False
export MIN_GRAPH_SIZE=0

PADDLE_TEST_BASE=./Paddle/test/dygraph_to_static
failed_tests=()
disabled_tests=(
    ${PADDLE_TEST_BASE}/test_lac.py # disabled by paddle
    ${PADDLE_TEST_BASE}/test_sentiment.py # disabled unitcase by paddle
    ${PADDLE_TEST_BASE}/test_pylayer.py # This ut cannot directly run
)

for file in ${PADDLE_TEST_BASE}/*.py; do
    # 检查文件是否为 Python 文件
    if [[ -f "$file" && ! "${disabled_tests[@]}" =~ "$file" ]]; then
        if [[ -n "$GITHUB_ACTIONS" ]]; then
            echo ::group::Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=${STRICT_MODE} python " $file
        else
            echo Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=${STRICT_MODE} python " $file
        fi
        # 执行文件
        # python "$file" 2>&1 >>/home/data/output.txt
        python -u "$file"
        if [ $? -ne 0 ]; then
            echo "run $file failed"
            failed_tests+=("$file")
        else
            echo "run $file success"
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
