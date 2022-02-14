# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import logging


class Context(object):
    def __init__(self, enable_plugin=True):
        self.args = self.parse_args()
        self.envs = self.fetch_envs()
        self.node = self.fetch_node_info()
        self.logger = self.get_logger()
        self.events = []

        # global status flag
        self.running = True

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
            help="The master/rendezvous server, ip:port")

        base_group.add_argument(
            "--rank", type=int, default=-1, help="The peer rank")

        base_group.add_argument(
            "--log", type=str, default="INFO", help="Log level. Default INFO")

        base_group.add_argument(
            "--nproc_per_node",
            type=int,
            default=None,
            help="The number of processes to launch on a node."
            "In gpu training, it should be less or equal to the gpus number of you system(or you set by --gpus). And so each process can"
            " bound to one or average number of gpus.")

        base_group.add_argument(
            "--log_dir",
            type=str,
            default="log",
            help="The path for each process's log. Default --log_dir=log/")
        '''
        base_group.add_argument(
            "--backend",
            type=str,
            default=os.environ.get('PADDLE_DISTRI_BACKEND', 'auto'),
            help="Specifize the backend, can be gloo|nccl|bkcl|auto|hccl|heter. "
            "Default value is auto which perfers nccl or bkcl.")
        '''
        base_group.add_argument(
            "--mode",
            type=str,
            default="collective",
            help="run mode of job, can be:collective/ps/ps-heter")

        base_group.add_argument(
            "--id", type=str, default="default", help="job unique id")

        base_group.add_argument(
            "--gpus",
            type=str,
            default=None,
            help="It's for gpu training."
            "For example:"
            "--gpus=\"0,1,2,3\" will launch four training processes each bound to one gpu."
        )
        base_group.add_argument("--selected_gpus", dest="gpus")

        base_group.add_argument(
            "--xpus",
            type=str,
            default=None,
            help="It's for xpu training. For example: "
            "--xpus=\"0,1,2,3\" will launch four training processes each bound to one xpu."
        )
        base_group.add_argument("--selected_xpus", dest="xpus")

        base_group.add_argument(
            "--npus",
            type=str,
            default=None,
            help="It's for xpu training. For example: "
            "--npus=\"0,1,2,3\" will launch four training processes each bound to one npu."
        )
        base_group.add_argument("--selected_npus", dest="npus")

        base_group.add_argument(
            "training_script",
            type=str,
            help="The full path to the single GPU training "
            "program/script to be launched in parallel, "
            "followed by all the arguments for the "
            "training script")

        base_group.add_argument('training_script_args', nargs=REMAINDER)
        '''
        # Optional arguments for the launch helper
        # for collective
        collective_group = parser.add_argument_group("Collective Parameters")
        collective_group.add_argument(
            "--ips",
            type=str,
            default="127.0.0.1",
            help="Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..")
        collective_group.add_argument(
            "--cluster_topo_path",
            type=str,
            default=None,
            help="A json format file will be stored in this path which is used"
            "to represent the cluster topology information for auto parallel.")
        collective_group.add_argument(
            "--rank_mapping_path",
            type=str,
            default=None,
            help="A json format file will be stored in this path which is used"
            "to map processes to machines for auto parallel.")
        collective_group.add_argument(
            "--enable_auto_mapping",
            type=bool,
            default=False,
            help="Set true to enable the lazy launch for auto-parallel scenario."
        )
        '''

        ps_group = parser.add_argument_group("Parameter-Server Parameters")
        # for parameter server
        ps_group.add_argument(
            "--servers",
            type=str,
            default="",
            help="User defined servers ip:port")
        ps_group.add_argument(
            "--workers",
            type=str,
            default="",
            help="User defined workers ip:port")
        ps_group.add_argument(
            "--heter_workers",
            type=str,
            default="",
            help="User defined heter workers in each stage ip1:port1;ip2:port2")
        ps_group.add_argument(
            "--heter_devices",
            type=str,
            default="",
            help="User defined heter devices in each stage cpu;gpu;cpu")

        ps_group.add_argument(
            "--trainer_num", type=int, help="number of trainers")
        ps_group.add_argument(
            "--server_num", type=int, help="number of servers")
        ps_group.add_argument(
            "--heter_worker_num",
            type=str,
            help="number of heter_workers in each stage 1;2;3")
        ps_group.add_argument("--http_port", type=int, help="Gloo http Port")

        # parameter elastic mode
        elastic_group = parser.add_argument_group("Elastic Parameters")
        elastic_group.add_argument(
            "--elastic_server", type=str, help="etcd server host:port")
        elastic_group.add_argument(
            "--elastic_pre_hook", type=str, help="elastic pre_hook shell cmd")

        elastic_group.add_argument(
            "--np", type=int, help="number of peer, job pod/node number")
        elastic_group.add_argument(
            "--scale", type=int, default=0, help="scale np")
        elastic_group.add_argument(
            "--host", type=str, help="bind host, default to POD_IP env")
        elastic_group.add_argument(
            "--force", type=bool, default=False, help="update np force")

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

    def fetch_node_info(self):
        return Node()

    def get_logger(self, level=logging.INFO):
        logger = logging.getLogger("PADDLERUN")
        logger.setLevel(self.args.log.upper() or level)
        formatter = logging.Formatter(
            fmt='%(name)s %(levelname)s %(asctime)s %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger
