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

set -x
set +e

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"

# install lcov
if [ ! -f "/root/.cache/lcov-1.16.tar.gz" ];then
wget -P /home https://paddle-ci.cdn.bcebos.com/coverage/lcov-1.16.tar.gz --no-proxy --no-check-certificate || exit 101
cp /home/lcov-1.16.tar.gz /root/.cache/lcov-1.16.tar.gz
else
    cp /root/.cache/lcov-1.16.tar.gz /home/lcov-1.16.tar.gz
fi
tar -xf /home/lcov-1.16.tar.gz -C /
cd /lcov-1.16
echo "::group::Install lcov"
make install
echo "::endgroup::"

cd ${PADDLE_ROOT}/build

python ${PADDLE_ROOT}/ci/coverage_gcda_clean.py ${PR_ID} || exit 101
echo "::group::Run lcov"
lcov --ignore-errors gcov --capture -d ./ -o coverage.info --rc lcov_branch_coverage=0
echo "::endgroup::"

mkdir coverage_files


function gen_full_report() {
    lcov --extract coverage.info \
        "${PADDLE_ROOT}/paddle/fluid/framework/*" \
        "${PADDLE_ROOT}/paddle/fluid/imperative/*" \
        "${PADDLE_ROOT}/paddle/fluid/inference/*" \
        "${PADDLE_ROOT}/paddle/fluid/memory/*" \
        "${PADDLE_ROOT}/paddle/fluid/operators/*" \
        "${PADDLE_ROOT}/paddle/fluid/eager/*" \
        "${PADDLE_ROOT}/paddle/fluid/pir/*" \
        "${PADDLE_ROOT}/paddle/fluid/ir_adaptor/*" \
        "${PADDLE_ROOT}/paddle/phi/*" \
        "${PADDLE_ROOT}/paddle/pir/*" \
        "${PADDLE_ROOT}/paddle/ap/*" \
        "${PADDLE_ROOT}/paddle/common/*" \
        "${PADDLE_ROOT}/paddle/utils/*" \
        "${PADDLE_ROOT}/paddle/cinn/*" \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info

    lcov --remove coverage-full.info \
        "${PADDLE_ROOT}/paddle/fluid/framework/*_test*" \
        "${PADDLE_ROOT}/paddle/fluid/*/*test*" \
        "${PADDLE_ROOT}/paddle/fluid/*/*/*test*" \
        "${PADDLE_ROOT}/paddle/fluid/inference/tests/*" \
        "${PADDLE_ROOT}/paddle/fluid/inference/api/demo_ci/*" \
        "${PADDLE_ROOT}/paddle/fluid/eager/tests/*" \
        "${PADDLE_ROOT}/paddle/phi/tests/*" \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info
}

function gen_full_report_xpu() {
    lcov --extract coverage.info \
        "${PADDLE_ROOT}/paddle/fluid/operators/*xpu*" \
        "${PADDLE_ROOT}/paddle/phi/kernels/xpu/*" \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info

    lcov --remove coverage-full.info \
        "${PADDLE_ROOT}/paddle/fluid/framework/*_test*" \
        "${PADDLE_ROOT}/paddle/fluid/*/*test*" \
        "${PADDLE_ROOT}/paddle/fluid/*/*/*test*" \
        "${PADDLE_ROOT}/paddle/fluid/inference/tests/*" \
        "${PADDLE_ROOT}/paddle/fluid/inference/api/demo_ci/*" \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info
}

function gen_full_report_npu() {
    lcov --extract coverage.info \
        "${PADDLE_ROOT}/paddle/fluid/operators/*npu*" \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info

    lcov --remove coverage-full.info \
        "${PADDLE_ROOT}/paddle/fluid/framework/*_test*" \
        "${PADDLE_ROOT}/paddle/fluid/*/*test*" \
        "${PADDLE_ROOT}/paddle/fluid/*/*/*test*" \
        "${PADDLE_ROOT}/paddle/fluid/inference/tests/*" \
        "${PADDLE_ROOT}/paddle/fluid/inference/api/demo_ci/*" \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info
}

if [ ${WITH_XPU:-OFF} == "ON" ]; then
    gen_full_report_xpu || true
else
    echo "::group::Gen full report"
    gen_full_report || true  # coverage-full.info
    echo "::endgroup::"
fi

# mkdir coverage

if [ "${PR_ID}" != "" ]; then

    COVERAGE_DIFF_PATTERN="`python ${PADDLE_ROOT}/ci/coverage_pull_request.py files ${PR_ID}`"

    python ${PADDLE_ROOT}/ci/coverage_pull_request.py diff ${PR_ID} > git-diff.out
fi

lcov --extract coverage-full.info \
    ${COVERAGE_DIFF_PATTERN} \
    -o coverage-diff.info \
    --rc lcov_branch_coverage=0

python ${PADDLE_ROOT}/ci/coverage_diff.py coverage-diff.info git-diff.out > coverage-diff.tmp

mv -f coverage-diff.tmp coverage-diff.info

cp coverage-diff.info coverage_files

# python coverage

coverage combine $(ls python-coverage.data.*) || NO_PYTHON_COVERAGE_DATA=1

coverage xml -i -o python-coverage.xml || [[ "${NO_PYTHON_COVERAGE_DATA}" == "1" ]]

# sed -i "s#/mnt\/paddle#${PADDLE_ROOT//\//\\/}#g" python-coverage.xml

`$(python ${PADDLE_ROOT}/ci/coverage_python_coverage.py > python-coverage.info)` || [[ "${NO_PYTHON_COVERAGE_DATA}" == "1" ]]


function gen_python_full_report() {
    lcov --extract python-coverage.info \
        "${PADDLE_ROOT}/python/*" \
        -o python-coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f python-coverage-full.tmp python-coverage-full.info

    lcov --remove python-coverage-full.info \
        '/*/tests/*' \
        -o python-coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f python-coverage-full.tmp python-coverage-full.info
}

gen_python_full_report || true  # python-coverage-full.info


if [ "${GIT_PR_ID}" != "" ]; then
    COVERAGE_DIFF_PATTERN="`python ${PADDLE_ROOT}/ci/coverage_pull_request.py files ${GIT_PR_ID}`"

    python ${PADDLE_ROOT}/ci/coverage_pull_request.py diff ${GIT_PR_ID} > python-git-diff.out
fi

lcov --extract python-coverage-full.info \
    ${COVERAGE_DIFF_PATTERN} \
    -o python-coverage-diff.info \
    --rc lcov_branch_coverage=0

python ${PADDLE_ROOT}/ci/coverage_diff.py python-coverage-diff.info python-git-diff.out > python-coverage-diff.tmp

mv -f python-coverage-diff.tmp python-coverage-diff.info

cp python-coverage-diff.info coverage_files
