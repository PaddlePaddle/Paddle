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


def get_cloud_cluster(args_node_ips, args_port, selected_gpus):
    """
    args_node_ips:string, args_port: int, selected_gpus:list
    """
    node_ips = os.getenv("PADDLE_TRAINERS")
    assert node_ips is not None, "PADDLE_TRAINERS should not be None"

    node_ip = os.getenv("POD_IP")
    assert node_ip is not None, "POD_IP should not be None"

    node_rank = os.getenv("PADDLE_TRAINER_ID")
    assert node_rank is not None, "PADDLE_TRAINER_ID should not be None"

    node_ips = node_ips.split(",")
    num_nodes = len(node_ips)
    node_rank = int(node_rank)

    if args_node_ips != "127.0.0.1" and args_node_ips != ",".join(node_ips):
        logger.warning(
            "Please NOTE: When using paddlecloud, cluster_node_ips is \
automatically got from PADDLE_TRAINERS(multi nodes) or POD_IP(single node).\
Your input cluster_node_ips: {} doesn't equals to IPs: {} from \
paddlecloud environment.".format(args_node_ips, node_ips))

    # DISTRIBUTED_TRAINER_ENDPOINTS: new environment since paddlecloud 1.8.4
    trainer_endpoints = os.getenv("DISTRIBUTED_TRAINER_ENDPOINTS")
    if trainer_endpoints is None:
        started_port = args_port
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
        ports = [
            x for x in range(started_port, started_port + len(selected_gpus))
        ]
        trainer_endpoints = []
        for ip in node_ips:
            trainer_endpoints += ["%s:%d" % (ip, port) for port in ports]
    else:
        trainer_endpoints = trainer_endpoints.split(",")

    logger.debug("parsed from args: node_ips:{} \
        node_ip:{} node_rank:{} trainer_endpoints:{}"
                 .format(node_ips, node_ip, node_rank, trainer_endpoints))

    cluster, pod = get_cluster(node_ips, node_ip, trainer_endpoints,
                               selected_gpus)
    return cluster, cluster.pods[node_rank]


def get_trainers_num():
    return int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
