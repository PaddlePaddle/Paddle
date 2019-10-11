#!/bin/bash
unset https_proxy http_proxy

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
rm -f ${name}*.log

# start the unit test
run_time=$(( $TEST_TIMEOUT - 10 ))
echo "run_time: ${run_time}"
timeout -s SIGKILL ${run_time} python -u ${name}.py > ${name}_run.log 2>&1
exit_code=$?
if [[ $exit_code -eq 0 ]]; then
    exit 0
fi

echo "${name} faild with ${exit_code}"

# paddle log
echo "${name} log"
cat -n ${name}*.log

#display system context
for i in {1..2}; do 
    sleep 2 
    ps -ef | grep -E "(test_|_test)"

    if hash "nvidia-smi" > /dev/null; then
        nvidia-smi
    fi
done

#display /tmp/files
ls -l /tmp/paddle.*

exit 1
