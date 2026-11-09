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

source ${PADDLE_ROOT}/ci/coverage_common.sh

# install lcov
install_lcov

cd ${PADDLE_ROOT}/build

function init_gcov_tool() {
    local cxx_compiler=""
    local compiler_version=""
    local compiler_major=""
    local gcov_version=""
    local gcov_major=""
    local candidate=""
    local cxx_compiler_for_version=""
    local gcov_candidates=()

    GCOV_TOOL=""

    if [ -f CMakeCache.txt ]; then
        cxx_compiler=$(awk -F= '/^CMAKE_CXX_COMPILER:FILEPATH=/ {print $2; exit}' CMakeCache.txt)
    fi

    if [ -n "${cxx_compiler}" ] && [ -x "${cxx_compiler}" ]; then
        cxx_compiler_for_version="${cxx_compiler}"
    elif command -v c++ >/dev/null 2>&1; then
        cxx_compiler_for_version=$(command -v c++)
    fi

    if [ -n "${cxx_compiler_for_version}" ]; then
        compiler_version=$("${cxx_compiler_for_version}" -dumpfullversion -dumpversion 2>/dev/null | head -n 1)
        compiler_major=${compiler_version%%.*}
        gcov_candidates+=("gcov-${compiler_version}" "gcov-${compiler_major}")
    fi
    gcov_candidates+=(gcov)

    for candidate in "${gcov_candidates[@]}"; do
        if [ -n "${candidate}" ] && command -v "${candidate}" >/dev/null 2>&1; then
            GCOV_TOOL=$(command -v "${candidate}")
            break
        fi
    done

    if [ -z "${GCOV_TOOL}" ]; then
        echo "ERROR: no gcov executable found for coverage collection"
        exit 101
    fi

    gcov_version=$("${GCOV_TOOL}" --version 2>/dev/null | head -n 1 | grep -oE '[0-9]+([.][0-9]+)+' | tail -n 1)
    gcov_major=${gcov_version%%.*}

    echo "CXX compiler for coverage: ${cxx_compiler_for_version:-unknown}"
    if [ -n "${cxx_compiler}" ] && [ "${cxx_compiler}" != "${cxx_compiler_for_version}" ]; then
        echo "CMake CXX compiler from cache: ${cxx_compiler}"
    fi
    echo "CXX compiler version for coverage: ${compiler_version:-unknown}"
    echo "GCOV tool for coverage: ${GCOV_TOOL}"
    "${GCOV_TOOL}" --version

    if [ -n "${compiler_major}" ] && [ -n "${gcov_major}" ] && [ "${compiler_major}" != "${gcov_major}" ]; then
        echo "ERROR: gcov major version ${gcov_major} does not match CXX compiler major version ${compiler_major}"
        exit 101
    fi
}

# NOTE: This gcda pre-cleaning step keeps/removes files by mapping PR
# changed files to "*.gcda" paths. That is not always correct: for header-only
# changes, or changes whose coverage is recorded in other translation units,
# valid gcda may be removed before lcov capture. Keep the current behavior for
# now because removing this step may increase "lcov/gcov --capture -d" cost.
python ${PADDLE_ROOT}/ci/coverage_gcda_clean.py ${PR_ID} || exit 101
echo "::group::Run lcov"
init_gcov_tool
lcov --gcov-tool "${GCOV_TOOL}" --ignore-errors gcov --capture -d ./ -o coverage.info --rc lcov_branch_coverage=0
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

# Shard mode (COVERAGE_SHARD_ONLY=ON): each test shard produces only the full
# C++/python reports for the tests it ran and uploads them to CFS. The
# "Coverage collect" job merges all shard reports and does the diff/judge part.
if [ "${COVERAGE_SHARD_ONLY:-OFF}" == "ON" ]; then
    echo "::group::Gen python full report (shard ${SHARD_INDEX})"
    coverage combine $(ls python-coverage.data.*) || NO_PYTHON_COVERAGE_DATA=1
    coverage xml -i -o python-coverage.xml || [[ "${NO_PYTHON_COVERAGE_DATA}" == "1" ]]
    `$(python ${PADDLE_ROOT}/ci/coverage_python_coverage.py > python-coverage.info)` || [[ "${NO_PYTHON_COVERAGE_DATA}" == "1" ]]
    if [ "${NO_PYTHON_COVERAGE_DATA}" != "1" ]; then
        gen_python_full_report
    fi
    echo "::endgroup::"
    shard_dir=${CFS_DIR}/coverage_shard/${PR_ID}/${COMMIT_ID}/shard-${SHARD_INDEX}
    mkdir -p ${shard_dir}
    cp coverage-full.info ${shard_dir}/
    if [ "${NO_PYTHON_COVERAGE_DATA}" != "1" ]; then
        cp python-coverage-full.info ${shard_dir}/
    fi
    echo "Uploaded shard coverage to ${shard_dir}"
    exit 0
fi

# mkdir coverage

# python coverage
coverage combine $(ls python-coverage.data.*) || NO_PYTHON_COVERAGE_DATA=1

coverage xml -i -o python-coverage.xml || [[ "${NO_PYTHON_COVERAGE_DATA}" == "1" ]]

# sed -i "s#/mnt\/paddle#${PADDLE_ROOT//\//\\/}#g" python-coverage.xml

`$(python ${PADDLE_ROOT}/ci/coverage_python_coverage.py > python-coverage.info)` || [[ "${NO_PYTHON_COVERAGE_DATA}" == "1" ]]

gen_python_full_report || true  # python-coverage-full.info

gen_coverage_diff
