#!/bin/bash
unset https_proxy http_proxy

name=${TEST_TARGET_NAME}

if [[ ${name}"x" == "x" ]]; then
    echo "can't find ${name}, please set ${TEST_TARGET_NAME} first"
    exit 1
fi

# rm flag file
flag=${name}.flag
rm -f ${flag} ${name}*.log

# start the unit test
nohup python -u ${name}.py > ${name}_run.log 2>&1 &
pid=$!

# check if the flag is generated.
for i in {1..110}; do 
    sleep 3s

    if [[ -f "${flag}" ]];  then
        echo "${name} succeed."
        exit 0
    fi

    if ! ps | grep -q "${pid}"; then
        if [[ ! -f "${flag}" ]];  then
            echo "${name} failed."
            exit 1
        fi
    fi
done

# paddle log
echo "${name}_run log"
cat -n ${name}*.log

#display system context
for i in {1..4}; do 
    sleep 2 
    ps -ef | grep -E "(test_|_test)"

    if hash "nvidia-smi" > /dev/null; then
        nvidia-smi
    fi
done

#display /tmp/files
ls -l /tmp/paddle.*

if ! pgrep -x ${name}; then
    exit 1
fi

kill -9 $pid

echo "after kill ${pid}"

#display system context
for i in {1..4}; do 
    sleep 2 
    ps -ef | grep -E "(test_|_test)"

    if hash "nvidia-smi" > /dev/null; then
        nvidia-smi
    fi
done

exit 1
