# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import paddle
from paddle.distributed.utils import get_cluster, logger


def get_cloud_cluster(args_node_ips, args_node_ip, args_port, selected_gpus):
    """
    args_node_ips, args_node_ip:string
    """
    #you can automatically get ip info while using paddlecloud multi nodes mode.
    node_ips = os.getenv("PADDLE_TRAINERS")
    assert node_ips is not None, "PADDLE_TRAINERS should not be None"

    node_ip = os.getenv("POD_IP")
    assert node_ip is not None, "POD_IP should not be None"

    node_rank = os.getenv("PADDLE_TRAINER_ID")
    assert node_rank is not None, "PADDLE_TRAINER_ID should not be None"

    node_ips = node_ips.split(",")
    num_nodes = len(node_ips)
    node_rank = int(node_rank)

    if node_ip != "127.0.0.1" and node_ip != args_node_ip:
        logger.warning("Please NOTE: When using paddlecloud, node_ip is \
automatically got from POD_IP. Your input node_ip: {} doesn't equals to \
node_ip: {} from paddlecloud environment.".format(args_node_ip, node_ip))

    if args_node_ips != "127.0.0.1" and args_node_ips != ",".join(node_ips):
        logger.warning(
            "Please NOTE: When using paddlecloud, cluster_node_ips is \
automatically got from PADDLE_TRAINERS(multi nodes) or POD_IP(single node).\
Your input cluster_node_ips: {} doesn't equals to IPs: {} from \
paddlecloud environment.".format(args_node_ips, node_ips))

    started_port = args_port
    print("num_nodes:", num_nodes)
    if num_nodes > 1:
        try:
            paddle_port = int(os.getenv("PADDLE_PORT", ""))
            paddle_port_num = int(os.getenv("TRAINER_PORTS_NUM", ""))

            if paddle_port_num >= len(
                    selected_gpus) and paddle_port != args_port:
                logger.warning("Use Cloud specified port:{}.".format(
                    paddle_port))
                started_port = paddle_port

        except Exception as e:
            print(e)
            pass

    if started_port is None:
        started_port = 6170

    logger.debug("parsed from args:node_ips:{} \
        node_ip:{} node_rank:{} started_port:{}"
                 .format(node_ips, node_ip, node_rank, started_port))

    ports = [x for x in range(started_port, started_port + len(selected_gpus))]
    cluster, pod = get_cluster(node_ips, node_ip, ports, selected_gpus)
    return cluster, cluster.pods[node_rank]


def get_trainers_num():
    return int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
