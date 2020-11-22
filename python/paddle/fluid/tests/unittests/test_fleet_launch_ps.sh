#!/bin/bash
set -e

function test_launch_ps(){
    fleetrun --server_num=2 --worker_num=2 fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi

    fleetrun --servers="127.0.0.1:6780,127.0.0.1:6781" --workers="127.0.0.1:6782,127.0.0.1:6783" fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi

    fleetrun --servers="127.0.0.1:6780,127.0.0.1:6781" --workers="127.0.0.1,127.0.0.1" fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi
}

function test_launch_ps_heter(){
    fleetrun --server_num=2 --worker_num=2 --heter_worker_num=2 fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test heter pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi
}

if [[ ${WITH_GPU} == "OFF" ]]; then
    echo "in cpu test mode"
    test_launch_ps
    exit 0
fi

echo "No.1 unittest"
test_launch_ps
test_launch_ps_heter