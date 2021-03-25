#!/bin/bash
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
