#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

set -e

#XDNN_PATH=/opt/xdnn # <path_to_xdnn>
#XRE_PATH=/opt/xre # <path_to_xre>

if [[ "$XDNN_PATH" == "" ]] || [[ ! -d "$XDNN_PATH" ]]; then
 echo "XDNN_PATH not set, or directory ${XDNN_PATH} not found, please export XDNN_PATH=<path_to_xdnn>."
 exit -1
fi

if [[ "$XRE_PATH" == "" ]] || [[ ! -d "$XRE_PATH" ]]; then
 echo "XRE_PATH not set, or directory ${XRE_PATH} not found, please export XRE_PATH=<path_to_xre>."
 exit -1
fi

#:<<!
export GLOG_v=0
export XPU_VISIBLE_DEVICES=0;
export XPUAPI_DEBUG=1;
export LD_LIBRARY_PATH=$XDNN_PATH/so:$XRE_PATH/so:$LD_LIBRARY_PATH

chmod +x ./build/example
./build/example
#!

:<<!
SSH_IP_ADDR=localhost
SSH_PORT=9031
SSH_USR_ID=root
SSH_USR_PWD=root

WORK_SPACE="/var/tmp/example"
EXPORT_ENVIRONMENT_VARIABLES="export GLOG_v=0;export XPU_VISIBLE_DEVICES=0;export XPUAPI_DEBUG=1;"
EXPORT_ENVIRONMENT_VARIABLES="${EXPORT_ENVIRONMENT_VARIABLES}export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH;"

sshpass -p $SSH_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_PORT $SSH_USR_ID@$SSH_IP_ADDR "rm -rf $WORK_SPACE"
sshpass -p $SSH_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_PORT $SSH_USR_ID@$SSH_IP_ADDR "mkdir -p $WORK_SPACE"
sshpass -p $SSH_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_PORT $XDNN_PATH/so/* $SSH_USR_ID@$SSH_IP_ADDR:$WORK_SPACE
sshpass -p $SSH_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_PORT $XRE_PATH/so/* $SSH_USR_ID@$SSH_IP_ADDR:$WORK_SPACE
sshpass -p $SSH_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_PORT ../build/libxpuplugin.so $SSH_USR_ID@$SSH_IP_ADDR:$WORK_SPACE
sshpass -p $SSH_USR_PWD scp -v -r -o ConnectTimeout=60 -o StrictHostKeyChecking=no -P $SSH_PORT build/example $SSH_USR_ID@$SSH_IP_ADDR:$WORK_SPACE
sshpass -p $SSH_USR_PWD ssh -v -o ConnectTimeout=60 -o StrictHostKeyChecking=no -p $SSH_PORT $SSH_USR_ID@$SSH_IP_ADDR "cd $WORK_SPACE; ${EXPORT_ENVIRONMENT_VARIABLES} chmod +x ./example; ./example"
!
