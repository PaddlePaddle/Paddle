#!/usr/bin/env bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

set -xe

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"

function lcov_init(){
    # install lcov
    if [ ! -f "/root/.cache/lcov-1.14.tar.gz" ];then
        wget -P /home https://paddle-ci.gz.bcebos.com/coverage/lcov-1.14.tar.gz --no-proxy --no-check-certificate || exit 101
        cp /home/lcov-1.14.tar.gz /root/.cache/lcov-1.14.tar.gz
    else
        cp /root/.cache/lcov-1.14.tar.gz /home/lcov-1.14.tar.gz
    fi
    tar -xf /home/lcov-1.14.tar.gz -C /
    cd /lcov-1.14
    make install
    cd -
}

function gen_cpp_covinfo(){
    # run paddle coverage
    cd /paddle/build
    python ${PADDLE_ROOT}/tools/coverage/gcda_clean.py ${GIT_PR_ID} || exit 101
    lcov --capture -d ./ -o coverage.info --rc lcov_branch_coverage=0
}


# full html report

function gen_full_html_report() {
    lcov --extract coverage.info \
        '/paddle/paddle/fluid/framework/*' \
        '/paddle/paddle/fluid/imperative/*' \
        '/paddle/paddle/fluid/inference/*' \
        '/paddle/paddle/fluid/memory/*' \
        '/paddle/paddle/fluid/operators/*' \
        '/paddle/paddle/fluid/recordio/*' \
        '/paddle/paddle/fluid/eager/*' \
        '/paddle/paddle/phi/*' \
        '/paddle/paddle/utils/*' \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info

    lcov --remove coverage-full.info \
        '/paddle/paddle/fluid/framework/*_test*' \
        '/paddle/paddle/fluid/*/*test*' \
        '/paddle/paddle/fluid/*/*/*test*' \
        '/paddle/paddle/fluid/inference/tests/*' \
        '/paddle/paddle/fluid/inference/api/demo_ci/*' \
        '/paddle/paddle/fluid/eager/tests/*' \
        '/paddle/paddle/phi/tests/*' \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info
}

function gen_full_html_report_xpu() {
    lcov --extract coverage.info \
        '/paddle/paddle/fluid/operators/*xpu*' \
        '/paddle/paddle/phi/kernels/xpu/*' \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info

    lcov --remove coverage-full.info \
        '/paddle/paddle/fluid/framework/*_test*' \
        '/paddle/paddle/fluid/*/*test*' \
        '/paddle/paddle/fluid/*/*/*test*' \
        '/paddle/paddle/fluid/inference/tests/*' \
        '/paddle/paddle/fluid/inference/api/demo_ci/*' \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info
}

function gen_full_html_report_npu() {
    lcov --extract coverage.info \
        '/paddle/paddle/fluid/operators/*npu*' \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info

    lcov --remove coverage-full.info \
        '/paddle/paddle/fluid/framework/*_test*' \
        '/paddle/paddle/fluid/*/*test*' \
        '/paddle/paddle/fluid/*/*/*test*' \
        '/paddle/paddle/fluid/inference/tests/*' \
        '/paddle/paddle/fluid/inference/api/demo_ci/*' \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info
}

# if [ ${WITH_XPU:-OFF} == "ON" ]; then
#     gen_full_html_report_xpu || true
# else
#     gen_full_html_report || true
# fi

# diff html report

function gen_diff_html_report() {
    if [ "${GIT_PR_ID}" != "" ]; then

        COVERAGE_DIFF_PATTERN="`python ${PADDLE_ROOT}/tools/coverage/pull_request.py files ${GIT_PR_ID}`"

        python ${PADDLE_ROOT}/tools/coverage/pull_request.py diff ${GIT_PR_ID} > git-diff.out
    fi

    lcov --extract coverage-full.info \
        ${COVERAGE_DIFF_PATTERN} \
        -o coverage-diff.info \
        --rc lcov_branch_coverage=0

    python ${PADDLE_ROOT}/tools/coverage/coverage_diff.py coverage-diff.info git-diff.out > coverage-diff.tmp

    mv -f coverage-diff.tmp coverage-diff.info

    genhtml -o coverage-diff -t 'Diff Coverage' --no-function-coverage --no-branch-coverage coverage-diff.info
}

# gen_diff_html_report || true

function gen_py_covinfo(){
    # python coverage

    export COVERAGE_FILE=/paddle/build/python-coverage.data
    coverage combine $(ls python-coverage.data.*) || NO_PYTHON_COVERAGE_DATA=1
    $(coverage xml -i -o python-coverage.xml) || [[ "${NO_PYTHON_COVERAGE_DATA}" == "1" ]]
    sed -i 's/mnt\/paddle/paddle/g' python-coverage.xml
    $(python ${PADDLE_ROOT}/tools/coverage/python_coverage.py > python-coverage.info) || [[ "${NO_PYTHON_COVERAGE_DATA}" == "1" ]]
}


# python full html report
#
function gen_python_full_html_report() {
    lcov --extract python-coverage.info \
        '/paddle/python/*' \
        -o python-coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f python-coverage-full.tmp python-coverage-full.info

    lcov --remove python-coverage-full.info \
        '/*/tests/*' \
        -o python-coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f python-coverage-full.tmp python-coverage-full.info
}

# gen_python_full_html_report || true

# python diff html report

function gen_python_diff_html_report() {
    if [ "${GIT_PR_ID}" != "" ]; then
        COVERAGE_DIFF_PATTERN="`python ${PADDLE_ROOT}/tools/coverage/pull_request.py files ${GIT_PR_ID}`"

        python ${PADDLE_ROOT}/tools/coverage/pull_request.py diff ${GIT_PR_ID} > python-git-diff.out
    fi

    lcov --extract python-coverage-full.info \
        ${COVERAGE_DIFF_PATTERN} \
        -o python-coverage-diff.info \
        --rc lcov_branch_coverage=0

    python ${PADDLE_ROOT}/tools/coverage/coverage_diff.py python-coverage-diff.info python-git-diff.out > python-coverage-diff.tmp

    mv -f python-coverage-diff.tmp python-coverage-diff.info

    genhtml -o python-coverage-diff \
        -t 'Python Diff Coverage' \
        --no-function-coverage \
        --no-branch-coverage \
        --ignore-errors source \
        python-coverage-diff.info
}

# gen_python_diff_html_report || true

# assert coverage lines

function covinfo_combine_full(){
    if [ -f "other-coverage.info" ] && [ -s "other-coverage.info" ];then
        if [ -f "infer-coverage.info" ] && [ -s "infer-coverage.info" ];then
            lcov -a other-coverage.info -a infer-coverage.info -o coverage.info
        else
            mv other-coverage.info coverage.info
        fi
    elif [ -f "infer-coverage.info" ] && [ -s "infer-coverage.info" ];then
        mv infer-coverage.info coverage.info
    else
        echo "Cannot found coverage.info"
    fi

    if [ -f "other-python-coverage.info" ] && [ -s "other-python-coverage.info" ];then
        if [ -f "infer-python-coverage.info" ] && [ -s "other-python-coverage.info" ];then
            lcov -a other-python-coverage.info -a infer-python-coverage.info -o python-coverage.info
        else
            mv other-python-coverage.info python-coverage.info
        fi
    elif [ -f "infer-python-coverage.info" ] && [ -s "infer-python-coverage.info" ];then
        mv infer-python-coverage.info python-coverage.info
    else
        echo "Cannot found python coverage.info"
    fi
    gen_python_full_html_report || true
    gen_full_html_report || true
}

function cov_rate_judge(){
    echo "Assert CPP Diff Coverage"
    python ${PADDLE_ROOT}/tools/coverage/coverage_lines.py coverage-diff.info 0.9 || COVERAGE_LINES_ASSERT=1

    echo "Assert Python Diff Coverage"

    if [ ${WITH_XPU:-OFF} == "ON" ]; then
        echo "XPU has no python coverage!"
    else
        if [[ python-coverage-diff.info ]];then
            python ${PADDLE_ROOT}/tools/coverage/coverage_lines.py python-coverage-diff.info 0.9 || PYTHON_COVERAGE_LINES_ASSERT=1
        fi
    fi
    if [ "$COVERAGE_LINES_ASSERT" = "1" ] || [ "$PYTHON_COVERAGE_LINES_ASSERT" = "1" ]; then
        echo "exit 9" > /tmp/paddle_coverage.result
        exit 9
    fi
}

function print_usage() {
    echo -e "\n${RED}Usage${NONE}:
    ${BOLD}${SCRIPT_NAME}${NONE} [OPTION]"

    echo -e "\n${RED}Options${NONE}:
    ${BLUE}gen_cov_info${NONE}: generate coverage info
    ${BLUE}test${NONE}: coverage info combine
    "
}

function main () {
    local CMD=$1
    lcov_init
    case $CMD in
      gen_cov_info)
        gen_cpp_covinfo
        gen_py_covinfo
        ;;
      combine_cov_info)
      covinfo_combine_full
      gen_diff_html_report || true
      gen_python_diff_html_report || true
      cov_rate_judge
        ;;
      *)
        print_usage
        exit 1
        ;;
      esac
}

main $@
