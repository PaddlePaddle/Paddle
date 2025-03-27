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

function run_sot_test() {
    PY_VERSION=$1
    PYTHON_WITH_SPECIFY_VERSION=python$PY_VERSION
    PY_VERSION_NO_DOT=$(echo $PY_VERSION | sed 's/\.//g')

    export STRICT_MODE=1
    export MIN_GRAPH_SIZE=0
    export SOT_LOG_LEVEL=0
    export FLAGS_cudnn_deterministic=True
    export SOT_ENABLE_STRICT_GUARD_CHECK=True

    # Install PaddlePaddle
    echo "::group::Installing paddle wheel..."
    $PYTHON_WITH_SPECIFY_VERSION -m pip install ${PADDLE_ROOT}/dist/paddlepaddle-0.0.0-cp${PY_VERSION_NO_DOT}-cp${PY_VERSION_NO_DOT}-linux_x86_64.whl
    echo "::endgroup::"
    # cd to sot test dir
    cd $PADDLE_ROOT/test/sot/

    # Run unittest
    failed_tests=()

    # Skip single tests that are currently not supported
    declare -a skip_files
    skiplist_filename="./skip_files_py$PY_VERSION_NO_DOT"
    if [ -f "$skiplist_filename" ];then
        # Prevent missing lines
        echo "" >> "$skiplist_filename"
        while IFS= read -r line; do
            skip_files+=("$line")
            echo "$line"
        done < "$skiplist_filename"
    else
        skip_files=()
    fi

    for file in ./test_*.py; do
        # check file is python file
        if [ -f "$file" ]; then
            if [[ "${skip_files[*]}"  =~ "${file}" ]]; then
                echo "skip ${PY_VERSION_NO_DOT} ${file}"
                continue
            fi
            echo Running:" STRICT_MODE=1 MIN_GRAPH_SIZE=0 SOT_LOG_LEVEL=0 FLAGS_cudnn_deterministic=True SOT_ENABLE_STRICT_GUARD_CHECK=True python " $file
            # run unittests
            python_output=$($PYTHON_WITH_SPECIFY_VERSION $file 2>&1)

            if [ $? -ne 0 ]; then
                echo -e "       ${RED}run $file failed"
                failed_tests+=("$file")
                echo "${python_output}"
            fi
        fi
    done

    if [ ${#failed_tests[@]} -ne 0 ]; then
        echo -e "       ${RED}failed tests file:${NONE}"
        for failed_test in "${failed_tests[@]}"; do
            echo -e "             ${RED}${failed_test}${NONE}"
        done
        exit 1
    fi
}

source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/utils.sh
init

run_sot_test $1
