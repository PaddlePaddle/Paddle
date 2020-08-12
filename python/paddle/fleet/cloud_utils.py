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
from paddle.fleet.launch_utils import get_cluster, logger


def get_cloud_cluster(args_node_ips, selected_gpus):
    """
    args_node_ips, args_node_ip:string
    """
    #you can automatically get ip info while using paddlecloud multi nodes mode.
    pdc_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
    assert pdc_endpoints is not None, "PADDLE_TRAINER_ENDPOINTS should not be None"

    node_ips = os.getenv("PADDLE_TRAINERS")
    assert node_ips is not None, "PADDLE_TRAINERS should not be None"

    node_ip = os.getenv("POD_IP")
    assert node_ip is not None, "POD_IP should not be None"

    node_rank = os.getenv("PADDLE_TRAINER_ID")
    assert node_rank is not None, "PADDLE_TRAINER_ID should not be None"

    trainer_num = os.getenv("TRAINER_PORTS_NUM")
    assert trainer_num is not None, "TRAINER_PORTS_NUM should not be None"

    node_ips = node_ips.split(",")
    node_rank = int(node_rank)
    pdc_endpoints = pdc_endpoints.split(",")
    trainer_num = int(trainer_num)
    trainer_endpoints = []
    # paddlecloud single-node show all ports num: PADDLE_TRAINER_ENDPOINTS={ip1:port1,...ip1:port8}
    # paddlecloud multi-node only concat fisrt port: PADDLE_TRAINER_ENDPOINTS={ip1:port1,ip2:port2,ip3:port3}
    if len(pdc_endpoints) == len(node_ips):
        start_ips = [x.strip().split(":")[0] for x in pdc_endpoints]
        assert start_ips == node_ips, "Ips resolved from PADDLE_TRAINER_ENDPOINTS:{} "\
                "and PADDLE_TRAINERS:{} are different. Please Check.".format(pdc_endpoints, node_ips)
        start_ports = [int(x.strip().split(":")[1]) for x in pdc_endpoints]
        for n_rank, start_port in enumerate(start_ports):
            free_ports = [
                x for x in range(start_port, start_port + len(selected_gpus))
            ]
            trainer_endpoints += [
                "%s:%d" % (start_ips[n_rank], port) for port in free_ports
            ]
    elif len(pdc_endpoints) == trainer_num * len(node_ips):
        trainer_endpoints = pdc_endpoints
    else:
        logger.fatal("PADDLE_TRAINER_ENDPOINTS:{} error and exit".format(
            pdc_endpoints))
        exit(1)

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


def use_paddlecloud():
    node_ips = os.getenv("PADDLE_TRAINERS")
    node_ip = os.getenv("POD_IP")
    node_rank = os.getenv("PADDLE_TRAINER_ID")
    trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
    if node_ips is None or node_ip is None or node_rank is None or trainer_endpoints is None:
        return False
    else:
        return True


def get_trainers_num():
    return int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
