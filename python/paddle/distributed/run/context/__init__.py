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

from argparse import ArgumentParser, REMAINDER
import os, copy

from paddle.distributed.run import plugins

from .node import Node
from .status import Status

import logging


class Context(object):
    def __init__(self, enable_plugin=True):
        self.args = self.parse_args()
        self.envs = self.fetch_envs()
        self.logger = self.get_logger()

        self.node = Node()
        self.status = Status()

        self.set_env_in_args()

        # design for event queue, later
        self.events = []

        if enable_plugin:
            self._enable_plugin()

    def get_envs(self):
        return self.envs.copy()

    def _enable_plugin(self):
        for pl in plugins.enabled_plugins:
            pl(self)

    def parse_args(self):
        parser = ArgumentParser()

        base_group = parser.add_argument_group("Base Parameters")

        base_group.add_argument(
            "--master",
            type=str,
            default=None,
            help="the master/rendezvous server, ip:port")

        base_group.add_argument(
            "--rank", type=int, default=-1, help="the peer rank")

        base_group.add_argument(
            "--log", type=str, default="INFO", help="log level. Default INFO")

        base_group.add_argument(
            "--np",
            type=str,
            default="1",
            help="the number of peers, i.e. pod/node number")

        base_group.add_argument(
            "--nproc_per_node",
            type=int,
            default=None,
            help="the number of processes in a pod")

        base_group.add_argument(
            "--log_dir",
            type=str,
            default="log",
            help="the path for each process's log. Default ./log")
        base_group.add_argument(
            "--mode",
            type=str,
            default="collective",
            help="run mode of the job, collective/ps/ps-heter")

        base_group.add_argument(
            "--id",
            type=str,
            default="default",
            help="unique id of the job. Default default")

        base_group.add_argument(
            "--devices",
            type=str,
            default=None,
            help="accelerate devices. as --gpus,npus,xps")

        base_group.add_argument(
            "--host", type=str, default=None, help="host ip")

        base_group.add_argument(
            "training_script",
            type=str,
            help="the full path of py script,"
            "followed by arguments for the "
            "training script")

        base_group.add_argument('training_script_args', nargs=REMAINDER)

        ps_group = parser.add_argument_group("Parameter-Server Parameters")
        # for parameter server
        ps_group.add_argument(
            "--servers",
            type=str,
            default='',
            help="servers endpoints full list")
        ps_group.add_argument(
            "--trainers",
            type=str,
            default='',
            help="trainers endpoints full list")

        ps_group.add_argument(
            "--trainer_num", type=int, default=None, help="number of trainers")
        ps_group.add_argument(
            "--server_num", type=int, default=None, help="number of servers")
        ps_group.add_argument(
            "--gloo_port", type=int, default=6767, help="gloo http port")
        ps_group.add_argument(
            "--with_gloo", type=str, default="0", help="use gloo or not")

        # parameter elastic mode
        elastic_group = parser.add_argument_group("Elastic Parameters")
        elastic_group.add_argument(
            "--max_restart",
            type=int,
            default=3,
            help="the times can restart. Default 3")

        elastic_group.add_argument(
            "--elastic_level",
            type=int,
            default=-1,
            help="elastic level: -1 disable, 0 failed exit, peers hold, 1 interal restart"
        )

        elastic_group.add_argument(
            "--elastic_timeout",
            type=int,
            default=30,
            help="seconds to wait before elastic perform training")
        return parser.parse_args()

    def _valide_env(self, key):
        if key in ['POD_IP']:
            return True
        if key.endswith('_VISIBLE_DEVICES'):
            return True
        if key.startswith('PADDLE_'):
            return True

        return False

    def fetch_envs(self):
        ge = os.environ.copy()
        return {k: ge[k] for k in ge if self._valide_env(k)}

    def get_logger(self, level=logging.INFO):
        logger = logging.getLogger("PADDLERUN")
        logger.setLevel(self.args.log.upper() or level)
        formatter = logging.Formatter(
            fmt='%(name)s %(levelname)s %(asctime)s %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def set_env_in_args(self):
        env_args = {
            'POD_IP': 'host',
            'PADDLE_MASTER': 'master',
            'PADDLE_DEVICES': 'devices',
            'PADDLE_NP': 'np',
            'PADDLE_MODE': 'mode',
            'PADDLE_LOG': 'log',
            'PADDLE_NPROC_PER_NODE': 'nproc_per_node',
            'PADDLE_JOB_ID': 'id',
            'PADDLE_RANK': 'rank',
            'PADDLE_LOG_DIR': 'log_dir',
            'PADDLE_MAX_RESTlRT': 'max_restart',
            'PADDLE_ELASTIC_LEVEL': 'elastic_level',
            'PADDLE_ELASTIC_TIMEOUT': 'elastic_timeout',
            'PADDLE_SERVER_NUM': 'server_num',
            'PADDLE_TRAINER_NUM': 'trainer_num',
            'PADDLE_SERVERS_ENDPOINTS': 'servers',
            'PADDLE_TRAINERS_ENDPOINTS': 'trainers',
            'PADDLE_GLOO_PORT': 'gloo_port',
            'PADDLE_WITH_GLOO': 'with_gloo',
        }

        for k, v in env_args.items():
            if k in self.envs:
                setattr(self.args, v, self.envs[k])
