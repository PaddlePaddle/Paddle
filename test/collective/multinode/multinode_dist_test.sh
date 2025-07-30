#!/bin/bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

export MPIRUN=${EXE_MPIRUN:-""} # MPIRUN="mpirun"

name=${TEST_TARGET_NAME}
TEST_TIMEOUT=${TEST_TIMEOUT}

if [[ ${name}"x" == "x" ]]; then
    echo "can't find ${name}, please set ${TEST_TARGET_NAME} first"
    exit 1
fi

if [[ ${TEST_TIMEOUT}"x" == "x" ]]; then
    echo "can't find ${TEST_TIMEOUT}, please set ${TEST_TIMEOUT} first"
    exit 1
fi


# rm flag file
rm -f ${name}_*.log

# start the unit test
run_time=$(( $TEST_TIMEOUT - 10 ))
echo "run_time: ${run_time}"

if [[ ${WITH_COVERAGE} == "ON" ]]; then
    PYTHON_EXEC="python3 -u -m coverage run --branch -p "
else
    PYTHON_EXEC="python3 -u "
fi

unset PYTHONPATH
timeout -s SIGKILL ${run_time} ${MPIRUN} ${PYTHON_EXEC} ${name}.py > ${name}_run.log 2>&1
exit_code=$?
if [[ $exit_code -eq 0 ]]; then
    exit 0
fi

echo "${name} failed with ${exit_code}"

echo "after run ${name}"
ps -aux
netstat -anlp

# paddle log
echo "${name} log"
for log in `ls ${name}_*.log`
do
    printf "\ncat ${log}\n"
    cat -n ${log}
done

# check CUDA or ROCM env
GPU_SYS_INFO_CMD=nvidia-smi

which ${GPU_SYS_INFO_CMD}
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    GPU_SYS_INFO_CMD=rocm-smi
fi

which ${GPU_SYS_INFO_CMD}
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    echo "nvidia-smi or rocm-smi failed with ${exit_code}"
    exit ${exit_code}
fi

#display system context
for i in {1..2}; do
    sleep 3
    ps -aux
    netstat -anlp

    if hash "${GPU_SYS_INFO_CMD}" > /dev/null; then
        ${GPU_SYS_INFO_CMD}
    fi
done

echo "dist space:"
df -h

#display /tmp/files
echo "ls /tmp/paddle.*"
ls -l /tmp/paddle.*

echo "ls -l ./"
ls -l ./

exit 1
