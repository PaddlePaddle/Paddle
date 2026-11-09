#!/bin/bash
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
#
# Shared helpers for the coverage CI scripts. Source this file, it only
# defines functions and does not execute anything by itself.

# Install lcov from the cached tarball (shared CFS cache at /root/.cache).
function install_lcov() {
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
}

# Extract the python/* full report from python-coverage.info and strip tests.
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

# Produce coverage-diff.info (C++) and python-coverage-diff.info from the
# full reports, scoped to the files touched by the PR. Must run in build dir.
function gen_coverage_diff() {
    # C++ diff
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

    # python diff
    if [ "${GIT_PR_ID}" != "" ]; then
        COVERAGE_DIFF_PATTERN="`python ${PADDLE_ROOT}/ci/coverage_pull_request.py files ${GIT_PR_ID}`"

        python ${PADDLE_ROOT}/ci/coverage_pull_request.py diff ${GIT_PR_ID} > python-git-diff.out
    fi

    if [ "${NO_PYTHON_COVERAGE_DATA}" != "1" ]; then
        lcov --extract python-coverage-full.info \
            ${COVERAGE_DIFF_PATTERN} \
            -o python-coverage-diff.info \
            --rc lcov_branch_coverage=0

        python ${PADDLE_ROOT}/ci/coverage_diff.py python-coverage-diff.info python-git-diff.out > python-coverage-diff.tmp

        mv -f python-coverage-diff.tmp python-coverage-diff.info

        cp python-coverage-diff.info coverage_files
    fi
}
