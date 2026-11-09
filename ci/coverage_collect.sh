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

set -x
set +e

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"

source ${PADDLE_ROOT}/ci/coverage_common.sh

echo "::group::Install PyGithub"
pip install PyGithub
echo "::endgroup::"

# install lcov
install_lcov

cd ${PADDLE_ROOT}/build
mkdir -p coverage_files

SHARD_ROOT=${CFS_DIR}/coverage_shard/${PR_ID}/${COMMIT_ID}
echo "Merging shard coverage reports from ${SHARD_ROOT}"

# Verify that every shard uploaded its report. A missing shard would silently
# understate coverage and produce a misleading diff/judge result.
NUM_SHARDS=${NUM_SHARDS:-1}
missing_shards=""
for (( i = 0; i < NUM_SHARDS; i++ )); do
    if [ ! -f "${SHARD_ROOT}/shard-${i}/coverage-full.info" ]; then
        missing_shards="${missing_shards} shard-${i}"
    fi
done
if [ -n "${missing_shards}" ]; then
    echo "ERROR: missing shard coverage reports:${missing_shards}"
    exit 101
fi

# Merge C++ coverage: lcov --add-tracefile accepts one file per flag
lcov_args=()
for f in ${SHARD_ROOT}/shard-*/coverage-full.info; do
    [ -f "${f}" ] && lcov_args+=("-a" "${f}")
done
lcov "${lcov_args[@]}" -o coverage-full.info --rc lcov_branch_coverage=0

# Merge python coverage: add all shard python-coverage-full.info
python_lcov_args=()
for f in ${SHARD_ROOT}/shard-*/python-coverage-full.info; do
    [ -f "${f}" ] && python_lcov_args+=("-a" "${f}")
done
if [ ${#python_lcov_args[@]} -gt 0 ]; then
    lcov "${python_lcov_args[@]}" -o python-coverage-full.info --rc lcov_branch_coverage=0
else
    NO_PYTHON_COVERAGE_DATA=1
fi

# ---- C++/python diff part (shared with coverage_info.sh) ----
gen_coverage_diff

# Merge shard cost data into the canonical CFS path. Each shard only recorded
# the tests it actually ran, so concatenation is duplicate-free; dedupe anyway
# to be safe (keep the last record per test name, emit a single "---" trailer).
mkdir -p /home/data/cfs/coverage/${PR_ID}/${COMMIT_ID}
cat ${SHARD_ROOT}/shard-*/CTestCostData.txt 2>/dev/null | awk '
    /^[^ ]+ [0-9]/ { lines[$1] = $0; next }
    /^[[:space:]]*$/ { next }
    { trailer_seen = 1 }
    END {
        for (t in lines) print lines[t];
        if (trailer_seen) print "---";
    }
' > /home/data/cfs/coverage/${PR_ID}/${COMMIT_ID}/CTestCostData.txt
cp coverage_files/* /home/data/cfs/coverage/${PR_ID}/${COMMIT_ID}/
