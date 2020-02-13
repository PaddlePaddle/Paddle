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


def get_cloud_env(args):
    current_node_ip = args.node_ip
    node_ips = [x.strip() for x in args.cluster_node_ips.split(',')]
    node_id = node_ips.index(current_node_ip)

    trainer_nums = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
    if trainer_nums != 1:
        #you can automatically get ip info while using paddlecloud multi nodes mode.
        current_node_ip = os.getenv("POD_IP")
        assert current_node_ip is not None, "POD_IP should not be None"

        node_ips = os.getenv("PADDLE_TRAINERS")
        assert node_ips is not None, "PADDLE_TRAINERS should not be None"

        node_ips = node_ips.split(",")
        node_id = os.getenv("PADDLE_TRAINER_ID")
        assert node_id is not None, "PADDLE_TRAINER_ID should not be None"
        node_id = int(node_id)

        if args.node_ip != "127.0.0.1" and current_node_ip != args.node_ip:
            logger.warning(
                "Please NOTE: When using paddlecloud, current_node_ip is \
automatically got from POD_IP. Your input node_ip: {} doesn't equals to \
current_node_ip: {} from paddlecloud environment."
                .format(args.node_ip, current_node_ip))
        if args.cluster_node_ips != "127.0.0.1" and args.cluster_node_ips != ",".join(
                node_ips):
            logger.warning(
                "Please NOTE: When using paddlecloud, cluster_node_ips is \
automatically got from PADDLE_TRAINERS(multi nodes) or POD_IP(single node).\
Your input cluster_node_ips: {} doesn't equals to IPs: {} from \
paddlecloud environment.".format(args.cluster_node_ips, node_ips))

        if args.use_paddlecloud and num_nodes > 1:
            cloud_paddle_port = os.getenv("PADDLE_PORT", "")
            cloud_paddle_port_num = os.getenv("PADDLE_PORTS_NUM", "")
            if cloud_paddle_port != "" and cloud_paddle_port_num != "":
                cloud_paddle_port_num = int(cloud_paddle_port_num)
                if cloud_paddle_port_num >= selected_gpus_num:
                    args.started_port = int(cloud_paddle_port)
                    logger.warning("Use Cloud specified port:{}.".format(
                        cloud_paddle_port))

    trainers_endpoints = ""
    for ip in node_ips:
        for i in range(selected_gpus_num):
            if trainers_endpoints != "":
                trainers_endpoints += ","
            trainers_endpoints += "%s:%d" % (ip, args.started_port + i)

    nranks = num_nodes * selected_gpus_num

    if args.print_config:
        print("trainers_endpoints:", trainers_endpoints, ", node_id:", node_id,
              ", current_node_ip:", current_node_ip, ", num_nodes:", num_nodes,
              ", node_ips:", node_ips, ", nranks:", nranks)

    #return node_ips, current_node_ip, node_id
    return cluster
