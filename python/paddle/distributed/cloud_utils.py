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


def get_cloud_cluster(args_node_ips, selected_gpus):
    """
    args_node_ips, args_node_ip:string
    """
    #you can automatically get ip info while using paddlecloud multi nodes mode.
    trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
    assert trainer_endpoints is not None, "PADDLE_TRAINER_ENDPOINTS should not be None"

    node_ips = os.getenv("PADDLE_TRAINERS")
    assert node_ips is not None, "PADDLE_TRAINERS should not be None"

    node_ip = os.getenv("POD_IP")
    assert node_ip is not None, "POD_IP should not be None"

    node_rank = os.getenv("PADDLE_TRAINER_ID")
    assert node_rank is not None, "PADDLE_TRAINER_ID should not be None"

    node_ips = node_ips.split(",")
    node_rank = int(node_rank)
    trainer_endpoints = trainer_endpoints.split(",")

    if args_node_ips != "127.0.0.1" and args_node_ips != ",".join(node_ips):
        logger.warning(
            "Please NOTE: When using paddlecloud, cluster_node_ips is \
automatically got from PADDLE_TRAINERS(multi nodes) or POD_IP(single node).\
Your input cluster_node_ips: {} doesn't equals to IPs: {} from \
paddlecloud environment.".format(args_node_ips, node_ips))

    logger.info("parsed from args:node_ips:{}"
                "node_ip:{} node_rank:{} trainer_endpoints:{}"
                .format(node_ips, node_ip, node_rank, trainer_endpoints))

    # if selected_gpus < len(cur_node_endpoints) : get_cluster() will only append first gpus_num pod.
    cluster, pod = get_cluster(node_ips, node_ip, trainer_endpoints,
                               selected_gpus)
    return cluster, cluster.pods[node_rank]


def get_trainers_num():
    return int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
