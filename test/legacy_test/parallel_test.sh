#!/bin/bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

unset https_proxy http_proxy
export FLAGS_rpc_disable_reuse_port=1

name=${TEST_TARGET_NAME}
UnitTests=${UnitTests}
TEST_TIMEOUT=${TEST_TIMEOUT}

if [[ ${name}"x" == "x" ]]; then
    echo "can't find name, please set TEST_TARGET_NAME first"
    exit 1
fi

if [[ ${UnitTests}"x" == "x" ]]; then
    echo "can't find UnitTests, please set TEST_TARGET_NAME first"
    exit 1
fi

if [[ ${TEST_TIMEOUT}"x" == "x" ]]; then
    echo "can't find ${TEST_TIMEOUT}, please set ${TEST_TIMEOUT} first"
    exit 1
fi

if [[ ${WITH_COVERAGE} == "ON" ]]; then
    PYTHON_EXEC="python -u -m coverage run --branch -p "
else
    PYTHON_EXEC="python -u "
fi

run_time=$(( $TEST_TIMEOUT - 10 ))
echo "run_time: ${run_time}"
for ut in ${UnitTests}; do
    echo "start ${ut}"
    timeout -s SIGKILL ${run_time} ${PYTHON_EXEC} ./${ut}.py > ${ut}_run.log 2>&1 &
done

FAIL=0
for job in `jobs -p`
do
    echo "jobs -p result:" `jobs -p`
    echo $job
    wait $job || let FAIL=FAIL+1
done

echo "fail_num:" $FAIL

if [ "$FAIL" == "0" ];
then
    exit 0
else
    echo "FAIL! ($FAIL)"

    for ut in ${UnitTests}; do
        log=${ut}_run.log
        echo "cat ${log}"
        cat $log
    done

    exit 1
fi
