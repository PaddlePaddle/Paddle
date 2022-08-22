# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from argparse import ArgumentParser, REMAINDER

env_args_mapping = {
    'POD_IP': 'host',
    'PADDLE_MASTER': 'master',
    'PADDLE_DEVICES': 'devices',
    'PADDLE_NNODES': 'nnodes',
    'PADDLE_RUN_MODE': 'run_mode',
    'PADDLE_LOG_LEVEL': 'log_level',
    'PADDLE_NPROC_PER_NODE': 'nproc_per_node',
    'PADDLE_JOB_ID': 'job_id',
    'PADDLE_RANK': 'rank',
    'PADDLE_LOG_DIR': 'log_dir',
    'PADDLE_MAX_RESTART': 'max_restart',
    'PADDLE_ELASTIC_LEVEL': 'elastic_level',
    'PADDLE_ELASTIC_TIMEOUT': 'elastic_timeout',
    'PADDLE_SERVER_NUM': 'server_num',
    'PADDLE_TRAINER_NUM': 'trainer_num',
    'PADDLE_SERVERS_ENDPOINTS': 'servers',
    'PADDLE_TRAINERS_ENDPOINTS': 'trainers',
    'PADDLE_GLOO_PORT': 'gloo_port',
    'PADDLE_WITH_GLOO': 'with_gloo',
    'PADDLE_START_PORT': 'start_port',
    'PADDLE_IPS': 'ips',
}


def fetch_envs():
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)

    return os.environ.copy()


def parse_args():
    parser = ArgumentParser()

    base_group = parser.add_argument_group("Base Parameters")

    base_group.add_argument("--master",
                            type=str,
                            default=None,
                            help="the master/rendezvous server, ip:port")

    base_group.add_argument("--legacy",
                            type=bool,
                            default=False,
                            help="use legacy launch")

    base_group.add_argument("--rank",
                            type=int,
                            default=-1,
                            help="the node rank")

    base_group.add_argument("--log_level",
                            type=str,
                            default="INFO",
                            help="log level. Default INFO")

    base_group.add_argument("--nnodes",
                            type=str,
                            default="1",
                            help="the number of nodes, i.e. pod/node number")

    base_group.add_argument("--nproc_per_node",
                            type=int,
                            default=None,
                            help="the number of processes in a pod")

    base_group.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="the path for each process's log. Default ./log")
    base_group.add_argument("--run_mode",
                            type=str,
                            default=None,
                            help="run mode of the job, collective/ps/ps-heter")

    base_group.add_argument("--job_id",
                            type=str,
                            default="default",
                            help="unique id of the job. Default default")

    base_group.add_argument("--devices",
                            "--gpus",
                            "--npus",
                            "--xpus",
                            type=str,
                            default=None,
                            help="accelerate devices. as --gpus,npus,xpus")

    base_group.add_argument("--host", type=str, default=None, help="host ip")

    base_group.add_argument("--ips",
                            type=str,
                            default=None,
                            help="nodes ips, e.g. 10.10.1.1,10.10.1.2")

    base_group.add_argument("--start_port",
                            type=int,
                            default=6070,
                            help="fix port start with")

    base_group.add_argument("training_script",
                            type=str,
                            help="the full path of py script,"
                            "followed by arguments for the "
                            "training script")

    base_group.add_argument('training_script_args', nargs=REMAINDER)

    ps_group = parser.add_argument_group("Parameter-Server Parameters")
    # for parameter server
    ps_group.add_argument("--servers",
                          type=str,
                          default='',
                          help="servers endpoints full list")
    ps_group.add_argument("--trainers",
                          type=str,
                          default='',
                          help="trainers endpoints full list")

    ps_group.add_argument("--trainer_num",
                          type=int,
                          default=None,
                          help="number of trainers")
    ps_group.add_argument("--server_num",
                          type=int,
                          default=None,
                          help="number of servers")
    ps_group.add_argument("--gloo_port",
                          type=int,
                          default=6767,
                          help="gloo http port")
    ps_group.add_argument("--with_gloo",
                          type=str,
                          default="1",
                          help="use gloo or not")

    # parameter elastic mode
    elastic_group = parser.add_argument_group("Elastic Parameters")
    elastic_group.add_argument("--max_restart",
                               type=int,
                               default=3,
                               help="the times can restart. Default 3")

    elastic_group.add_argument(
        "--elastic_level",
        type=int,
        default=-1,
        help=
        "elastic level: -1 disable, 0 failed exit, peers hold, 1 internal restart"
    )

    elastic_group.add_argument(
        "--elastic_timeout",
        type=int,
        default=30,
        help="seconds to wait before elastic job begin to train")

    return parser.parse_known_args()
