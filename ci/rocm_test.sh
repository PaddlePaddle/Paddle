# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# ROCm GPU Test Script - based on coverage_test.sh

source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/utils.sh
init

# Set ROCm environment
export WITH_ROCM=ON
export WITH_TESTING=ON

mkdir -p ${PADDLE_ROOT}/build
cd ${PADDLE_ROOT}/build

echo "::group::Install dependencies"
pip install hypothesis 2>/dev/null || true
pip install -r ${PADDLE_ROOT}/python/unittest_py/requirements.txt 2>/dev/null || true
echo "::endgroup::"

echo "::group::Install paddle"
if ls ${PADDLE_ROOT}/build/python/dist/*whl >/dev/null 2>&1; then
    pip install ${PADDLE_ROOT}/build/python/dist/*whl --force-reinstall
elif ls ${PADDLE_ROOT}/dist/*whl >/dev/null 2>&1; then
    pip install ${PADDLE_ROOT}/dist/*whl --force-reinstall
fi
echo "::endgroup::"

# Copy test support files from source directory (not build directory)
# Note: coverage_test.sh has a bug copying empty file from build dir
cp ${PADDLE_ROOT}/test/legacy_test/testsuite.py ${PADDLE_ROOT}/build/test/legacy_test/ 2>/dev/null || true
cp ${PADDLE_ROOT}/test/legacy_test/testsuite.py ${PADDLE_ROOT}/build/python 2>/dev/null || true
cp -r ${PADDLE_ROOT}/build/test/white_list ${PADDLE_ROOT}/build/python 2>/dev/null || true

# Add source test directories to PYTHONPATH for module imports
export PYTHONPATH=${PADDLE_ROOT}/test:${PADDLE_ROOT}/test/legacy_test:${PYTHONPATH}

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
