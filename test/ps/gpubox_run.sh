# !/bin/bash

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

if [ ! -d "./log" ]; then
  mkdir ./log
  echo "Create log floder for store running log"
fi

export FLAGS_LAUNCH_BARRIER=0
export PADDLE_TRAINER_ID=0
export PADDLE_PSERVER_NUMS=1
export PADDLE_TRAINERS=1
export PADDLE_TRAINERS_NUM=${PADDLE_TRAINERS}
export POD_IP=127.0.0.1

# set free port if 29011 is occupied
export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:29011"
export PADDLE_PSERVER_PORT_ARRAY=(29011)

# set gpu numbers according to your device
#export FLAGS_selected_gpus="0,1,2,3,4,5,6,7"
export FLAGS_selected_gpus="0,1"

# set your model yaml
#SC="gpubox_ps_trainer.py"
SC="static_gpubox_trainer.py"

# run pserver
export TRAINING_ROLE=PSERVER
for((i=0;i<$PADDLE_PSERVER_NUMS;i++))
do
    cur_port=${PADDLE_PSERVER_PORT_ARRAY[$i]}
    echo "PADDLE WILL START PSERVER "$cur_port
    export PADDLE_PORT=${cur_port}
    python -u $SC &> ./log/pserver.$i.log &
done

# run trainer
export TRAINING_ROLE=TRAINER
for((i=0;i<$PADDLE_TRAINERS;i++))
do
    echo "PADDLE WILL START Trainer "$i
    export PADDLE_TRAINER_ID=$i
    python -u $SC &> ./log/worker.$i.log
done

if [ $? -eq 0 ];then
echo "Training log stored in ./log/"
else
exit 1
fi
