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

function is_run_distribute_in_op_test() {
    DISTRIBUTE_FILES=("python/paddle/distributed"
                      "python/phi/infermeta/spmd_rules"
                      "paddle/phi/core/distributed")
    cd ${PADDLE_ROOT}
    for DISTRIBUTE_FILE in ${DISTRIBUTE_FILES[*]}; do
        DISTRIBUTE_CHANGE=`git diff --name-only upstream/$BRANCH | grep -F "${DISTRIBUTE_FILE}"|| true`
        if [ "${DISTRIBUTE_CHANGE}" ] && [ "${GIT_PR_ID}" != "" ]; then
            export FLAGS_COVERAGE_RUN_AUTO_PARALLEL_IN_OP_TEST=1
            echo "export FLAGS_COVERAGE_RUN_AUTO_PARALLEL_IN_OP_TEST=1" >> "$HOME/.bashrc"
        fi
    done
    ALL_CHANGE_FILES=`git diff --numstat upstream/$BRANCH | awk '{print $3}' | grep ".py"|| true`
    echo ${ALL_CHANGE_FILES}
    for CHANGE_FILE in ${ALL_CHANGE_FILES}; do
        ALL_OPTEST_BAN_AUTO_PARALLEL_TEST=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${CHANGE_FILE} | grep "+" | grep "check_auto_parallel=" || true`
        if [ "${ALL_OPTEST_BAN_AUTO_PARALLEL_TEST}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
            export FLAGS_COVERAGE_RUN_AUTO_PARALLEL_IN_OP_TEST=1
            echo "export FLAGS_COVERAGE_RUN_AUTO_PARALLEL_IN_OP_TEST=1" >> "$HOME/.bashrc"
        fi
    done
}


ldconfig

echo "export PATH=/usr/local/bin:\${PATH}" >> ~/.bashrc
# echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.0/compat:\$LD_LIBRARY_PATH" >> ~/.bashrc

echo "alias grep='grep --color=auto'" >> "$HOME/.bashrc"
source ~/.bashrc
unset GREP_OPTIONS

source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/utils.sh
init

ln -sf $(which python3.9) /usr/local/bin/python
ln -sf $(which pip3.9) /usr/local/bin/pip

echo "::group::Install zstd"
apt install zstd -y
echo "::endgroup::"
pip config set global.cache-dir "/root/.cache/pip"
pip install --upgrade pip

echo "::group::Install dependencies"

pip install -r "${work_dir}/python/requirements.txt"
pip install -r "${work_dir}/python/unittest_py/requirements.txt"
pip install PyGithub

echo "::endgroup::"

is_run_distribute_in_op_test $1

mkdir -p ${PADDLE_ROOT}/build
cd ${PADDLE_ROOT}/build
echo "::group::Install hypothesis"
pip install hypothesis
echo "::endgroup::"
echo "::group::Install paddle"
if ls ${PADDLE_ROOT}/build/python/dist/*whl >/dev/null 2>&1; then
    pip install ${PADDLE_ROOT}/build/python/dist/*whl
fi
if ls ${PADDLE_ROOT}/dist/*whl >/dev/null 2>&1; then
    pip install ${PADDLE_ROOT}/dist/*whl
fi
echo "::endgroup::"
cp ${PADDLE_ROOT}/build/test/legacy_test/testsuite.py ${PADDLE_ROOT}/build/python
cp -r ${PADDLE_ROOT}/build/test/white_list ${PADDLE_ROOT}/build/python

ut_total_startTime_s=`date +%s`

parallel_test_base_gpu_test

ut_total_endTime_s=`date +%s`
echo "TestCases Total Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s"
echo "ipipe_log_param_TestCases_Total_Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt

if [[ -f ${PADDLE_ROOT}/build/build_summary.txt ]];then
echo "=====================build summary======================"
cat ${PADDLE_ROOT}/build/build_summary.txt
echo "========================================================"
fi
