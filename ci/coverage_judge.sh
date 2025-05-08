# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# assert coverage lines

echo "Assert Diff Coverage"
cd ${PADDLE_ROOT}/build

python ${PADDLE_ROOT}/tools/coverage/coverage_lines.py coverage-diff.info 0.9 || COVERAGE_LINES_ASSERT=1

echo "Assert Python Diff Coverage"

if [ ${WITH_XPU:-OFF} == "ON" ]; then
    echo "XPU has no python coverage!"
else
    if [[ "${NO_PYTHON_COVERAGE_DATA}" != "1" ]];then
        python ${PADDLE_ROOT}/tools/coverage/coverage_lines.py python-coverage-diff.info 0.9 || PYTHON_COVERAGE_LINES_ASSERT=1
    fi
fi

if [ "$COVERAGE_LINES_ASSERT" = "1" ] || [ "$PYTHON_COVERAGE_LINES_ASSERT" = "1" ]; then
    echo "exit 9" > /tmp/paddle_coverage.result
    python ${PADDLE_ROOT}/tools/get_pr_title.py skip_coverage_check && NOT_CHECK_COVERAGE_PR=1
    if [[ "${NOT_CHECK_COVERAGE_PR}" = "1" ]];then
        echo "Skip coverage check in the Coverage pipeline."
        exit 0
    fi
    exit 9
fi
