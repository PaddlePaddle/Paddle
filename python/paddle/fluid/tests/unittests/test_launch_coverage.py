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

from __future__ import print_function
import sys
import subprocess
import os
import time
import six
import copy
import unittest
import paddle.fluid as fluid

from argparse import ArgumentParser, REMAINDER
from paddle.distributed.utils import _print_arguments, get_gpus, get_cluster_from_args
from paddle.distributed.fleet.launch_utils import find_free_ports


def _parse_args():
    parser = ArgumentParser(
        description='''start paddle training using multi-process mode.
NOTE: your train program ***must*** run as distributed nccl2 mode,
see: http://www.paddlepaddle.org/documentation/docs/zh/1.6/user_guides/howto/training/cluster_howto.html#permalink-8--nccl2-
And your train program must read environment variables below in order to let different
process init properly:
FLAGS_selected_gpus
PADDLE_TRAINER_ID
PADDLE_CURRENT_ENDPOINT
PADDLE_TRAINERS_NUM
PADDLE_TRAINER_ENDPOINTS
POD_IP (current node ip address, not needed for local training)
''')

    #Optional arguments for the launch helper
    parser.add_argument(
        "--cluster_node_ips",
        type=str,
        default="127.0.0.1",
        help="Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..")
    parser.add_argument("--node_ip",
                        type=str,
                        default="127.0.0.1",
                        help="The current node ip. ")
    parser.add_argument(
        "--use_paddlecloud",
        action='store_true',
        help=
        "wheter to use paddlecloud platform to run your multi-process job. If false, no need to set this argument."
    )
    parser.add_argument("--started_port",
                        type=int,
                        default=None,
                        help="The trainer's started port on a single node")

    parser.add_argument("--print_config",
                        type=bool,
                        default=True,
                        help="Print the config or not")

    parser.add_argument(
        "--selected_gpus",
        type=str,
        default=None,
        help=
        "It's for gpu training and the training process will run on the selected_gpus,"
        "each process is bound to a single GPU. And if it's not set, this module will use all the gpu cards for training."
    )

    parser.add_argument(
        "--log_level",
        type=int,
        default=
        20,  # logging.INFO, details are here:https://docs.python.org/3/library/logging.html#levels
        help="Logging level, default is logging.INFO")

    parser.add_argument(
        "--log_dir",
        type=str,
        help=
        "The path for each process's log.If it's not set, the log will printed to default pipe."
    )

    #positional
    parser.add_argument("training_script",
                        type=str,
                        help="The full path to the single GPU training "
                        "program/script to be launched in parallel, "
                        "followed by all the arguments for the "
                        "training script")

    #rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


class TestCoverage(unittest.TestCase):

    def test_gpus(self):
        args = _parse_args()

        if args.print_config:
            _print_arguments(args)

        gpus = get_gpus(None)

        args.use_paddlecloud = True
        cluster, pod = get_cluster_from_args(args, "0")

    def test_find_free_ports(self):
        find_free_ports(2)


if __name__ == '__main__':
    unittest.main()
