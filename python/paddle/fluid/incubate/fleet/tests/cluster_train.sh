#!/bin/bash

# start pserver0
python fleet_deep_ctr.py \
    --role pserver \
    --endpoints 127.0.0.1:7000,127.0.0.1:7001 \
    --current_endpoint 127.0.0.1:7000 \
    --trainers 2 \
    > pserver0.log 2>&1 &

# start pserver1
python fleet_deep_ctr.py \
    --role pserver \
    --endpoints 127.0.0.1:7000,127.0.0.1:7001 \
    --current_endpoint 127.0.0.1:7001 \
    --trainers 2 \
    > pserver1.log 2>&1 &

# start trainer0
python fleet_deep_ctr.py \
    --role trainer \
    --endpoints 127.0.0.1:7000,127.0.0.1:7001 \
    --trainers 2 \
    --trainer_id 0 \
    > trainer0.log 2>&1 &

# start trainer1
python fleet_deep_ctr.py \
    --role trainer \
    --endpoints 127.0.0.1:7000,127.0.0.1:7001 \
    --trainers 2 \
    --trainer_id 1 \
    > trainer1.log 2>&1 &
