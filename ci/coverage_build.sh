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

source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/utils.sh
init

function check_diff_file_for_coverage() {
    diff_h_file=$(git diff --name-status test -- | awk '$1 != "D" {print $2}' | grep '\.h$' | awk -F "/" '{printf "%s,",$NF}')
    diff_cc_file=$(git diff --name-status test -- | awk '$1 != "D" {print $2}' | grep -E '\.(cc|c)$' | awk -F "/" '{printf "%s,",$NF}')
    diff_py_file=$(git diff --name-status test -- | grep '\.py$' | awk '$1 != "D" {printf "%s,",$2}')
    export PADDLE_GIT_DIFF_H_FILE=${diff_h_file%*,}
    export PADDLE_GIT_DIFF_CC_FILE=${diff_cc_file%*,}
    export PADDLE_GIT_DIFF_PY_FILE=${diff_py_file%*,}
    echo "export PADDLE_GIT_DIFF_H_FILE=${diff_h_file%*,}" >> ~/.bashrc
    echo "export PADDLE_GIT_DIFF_CC_FILE=${diff_cc_file%*,}" >> ~/.bashrc
    echo "export PADDLE_GIT_DIFF_PY_FILE=${diff_py_file%*,}" >> ~/.bashrc
}

echo "export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib:/usr/local/cuda/lib64:\\" >> ~/.bashrc
echo "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH}" >> ~/.bashrc
echo "export PATH=/usr/local/bin:\\" >> ~/.bashrc
echo "\${PATH}" >> ~/.bashrc
source ~/.bashrc

check_diff_file_for_coverage

bash $ci_scripts/run_setup.sh "$@"
