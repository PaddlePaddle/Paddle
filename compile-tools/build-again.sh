#!/bin/bash

export PADDLE_VERSION=2.2.1

cd build
use_proxy=0
if [ ! -n "$1" ] ;then
    use_proxy=$1
fi

while :
do
    source set_http_proxy.sh $use_proxy

    make -j 24
    if [ $? -eq 0 ]; then
        break
    fi
    echo "try to choose other proxy, compile again"
    if [ $use_proxy -eq 0 ]; then
        use_proxy=1
    else
        use_proxy=0
    fi
    sleep 3
done

make install
