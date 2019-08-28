#!/bin/bash
unset https_proxy http_proxy

nohup python -u test_listen_and_serv_op.py > test_listen_and_serv_op.log 2>&1 &
pid=$!

time_out=True
for i in {1..10}; do 
    sleep 3s
    if ! pgrep -x test_listen_and_serv_op; then
        time_out=False
        break
    fi
done

if [[ $time_out == "False" ]]; then
    echo "test_listen_and_serv_op exit"
    exit 0
fi

echo "test_listen_and_serv_op.log context"
cat test_listen_and_serv_op.log

#display system context
for i in {1..4}; do 
    sleep 2 
    top -b -n1  | head -n 50
    echo "${i}"
    top -b -n1 -i  | head -n 50
    nvidia-smi
done

#display /tmp/files
ls -l /tmp/paddle.*

kill -9 $pid

echo "after kill ${pid}"

#display system context
for i in {1..4}; do 
    sleep 2 
    top -b -n1  | head -n 50
    top -b -n1 -i  | head -n 50
    nvidia-smi
done

exit 1
