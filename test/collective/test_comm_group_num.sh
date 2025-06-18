# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

function is_a100() {
    if [ $(nvidia-smi|grep A100|wc -l)  -ne 0 ];then
        echo 1
    else
        echo 0
    fi
}

if [ "$(is_a100)" == "1" ]; then
    exit 0
fi

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_DEBUG=INFO
# export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22
export NCCL_IB_ADAPTIVE_ROUTING=0
# export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=0
export NCCL_SOCKET_IFNAME=xgbe0
# export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,ALLOC
export NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING,ALLOC
export NCCL_IB_HCA=mlx5_1,mlx5_8,mlx5_6,mlx5_4,mlx5_2,mlx5_9,mlx5_7,mlx5_5,mlx5_3
# export IB_GID_INDEX=3

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

START_RANK=0
END_RANK=1

if [[ $rank -lt $START_RANK ]]; then
    exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
    exit 0
fi
rank=$(($rank-$START_RANK))
nnodes=$(($END_RANK-$START_RANK))
# master=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
# master=`cat hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
if [ -f "/root/paddlejob/workspace/hostfile" ]; then
    # 文件存在，按原逻辑获取 master
    master=$(cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}')
else
    # 文件不存在，设置为当前机器的 IP
    master=$(hostname -I | awk '{print $1}')  # 获取本机 IP
    echo "hostfile not found, using current machine IP: $master"
fi
port=36677

version=2_21_5
new_api=4

if [ "$version" = "2_21_5" ]; then
    export NCCL_RUNTIME_CONNECT=0
fi

# root_path 改成自己的路径
root_path=/root/paddlejob/workspace/env_run/output/test_comm_group_num
task_name=llama2_13b_dynamic_hand_"$version"_"$new_api"
export NCCL_DEBUG_FILE=$root_path/Nccl/nccl_log/$task_name/%h.%p.log

export FLAGS_eager_communication_connection=1

export NNODES=1
export PADDLE_TRAINERS_NUM=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# log
export GLOG_v=3

log_dir="$root_path/Nccl/log/$task_name"

if [ -d $log_dir ]; then
    rm -rf $log_dir
fi

shell_dir=$(dirname "$(readlink -f "$0")")

python -u -m paddle.distributed.launch \
--gpus "0,1,2,3,4,5,6,7" \
--log_dir ${log_dir} \
--master $master:$port \
--nnodes $nnodes \
--rank $rank \
--run_mode=collective \
$shell_dir/test_comm_group_num.py

count7=$(grep -c "init NCCLCommContext rank" "${log_dir}/workerlog.7")

if [ $count7 -ne 7 ]; then
    echo -e "\033[31m test_comm_group_num failed, got ${count7}, expect 7 \033[0m"
    exit 1
fi

rm -rf $root_path
