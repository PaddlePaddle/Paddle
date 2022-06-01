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

import paddle.fluid as fluid

import subprocess
import argparse
import os
import logging
import sys


class IPULaunch(object):
    def __init__(self, hosts, ipus_per_replica, nproc_per_host, ipu_partition,
                 vipu_server, training_script, training_script_args):
        if not fluid.core.is_compiled_with_ipu():
            raise RuntimeError(
                "Can not call ipu_launch.py in non IPU compiled environment, please re-compile with WITH_IPU=ON."
            )
        self._hosts = hosts
        self._ipus_per_replica = ipus_per_replica
        self._nproc_per_host = nproc_per_host
        self._ipu_partition = ipu_partition
        self._vipu_server = vipu_server
        self._training_script = training_script
        self._training_script_args = training_script_args

        self._num_ipus = int(os.getenv("FLAGS_selected_ipus"))
        self.logger = self.get_logger()

    @classmethod
    def parse_ipu_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--hosts",
            type=str,
            help="The hosts for IPU PopRun distributd computing.")
        parser.add_argument(
            "--ipus_per_replica",
            type=int,
            help="The number of IPUs per replica.")
        parser.add_argument(
            "--nproc_per_host",
            type=int,
            help="The number of processes per host.")
        parser.add_argument(
            "--ipu_partition", type=str, help="The partition name of IPU.")
        parser.add_argument(
            "--vipu_server",
            type=str,
            help="The vipu server host to enable vipu.")
        parser.add_argument(
            "training_script",
            type=str,
            help="The full path to the single IPU replica training program/script to be launched in parallel."
        )
        parser.add_argument('training_script_args', nargs=argparse.REMAINDER)
        args = parser.parse_args()

        ipu_launch = IPULaunch(
            hosts=args.hosts,
            ipus_per_replica=args.ipus_per_replica,
            nproc_per_host=args.nproc_per_host,
            ipu_partition=args.ipu_partition,
            vipu_server=args.vipu_server,
            training_script=args.training_script,
            training_script_args=args.training_script_args, )

        return ipu_launch

    def get_logger(self, level=logging.INFO):
        logger = logging.getLogger("LAUNCH")
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt='%(name)s %(levelname)s %(asctime)s %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def launch(self):
        # The number of replicas for data parallel
        assert (self._num_ipus % self._ipus_per_replica) == 0, \
                    "The number of IPUs:{} mod the number of IPUs per replica:{} must == 0".format(self._num_ipus, self._ipus_per_replica)
        num_replicas = self._num_ipus // self._ipus_per_replica
        self.logger.info("The number of total replicas is {}.".format(
            num_replicas))

        # The number of processes
        num_nodes = len(self._hosts.split(','))
        num_procs = num_nodes * self._nproc_per_host
        self.logger.info("The number of total processes is {}.".format(
            num_procs))
        assert (num_replicas % num_procs) == 0, \
                    "The number of replicas:{} mod the number of processes:{} must == 0".format(num_replicas, num_procs)

        # hosts and endpoints
        hosts = self._hosts.replace(' ', '').split(',')
        endpoints = [x + ":8090" for x in hosts]

        # args for poprun
        poprun_command = ['poprun']

        poprun_command.append('--num-instances={}'.format(num_procs))
        poprun_command.append('--num-replicas={}'.format(num_replicas))
        poprun_command.append('--ipus-per-replica={}'.format(
            self._ipus_per_replica))
        poprun_command.append('--host={}'.format(','.join(hosts)))
        poprun_command.append('--vipu-partition={}'.format(self._ipu_partition))
        poprun_command.append('--vipu-server-host={}'.format(self._vipu_server))

        poprun_command.extend([
            '--update-partition=no', '--vipu-server-timeout=120',
            '--print-topology=yes', '--numa-aware=yes'
        ])

        # global envs
        global_envs = '--mpi-local-args=\''
        log_level = os.getenv('POPART_LOG_LEVEL', None)
        if log_level:
            global_envs += '-x POPART_LOG_LEVEL={} '.format(log_level)
        global_envs += '-x PADDLE_TRAINERS_NUM={} -x PADDLE_TRAINER_ENDPOINTS={}'.format(
            num_procs, ','.join(endpoints))
        global_envs += '\''
        poprun_command.append(global_envs)

        # local envs
        for idx in range(num_procs):
            cur_endpoint = endpoints[idx // self._nproc_per_host]
            rank_in_node = idx % self._nproc_per_host
            poprun_command.append(
                '--instance-mpi-local-args={}:\"-x PADDLE_TRAINER_ID={} -x PADDLE_CURRENT_ENDPOINT={} -x PADDLE_RANK_IN_NODE={}\"'.
                format(idx, idx, cur_endpoint, rank_in_node))

        # executor
        poprun_command.append(sys.executable)

        # script and script args
        poprun_command.append(self._training_script)
        for arg in self._training_script_args:
            poprun_command.append(arg)

        # for debug
        print("-----------  PopRun Command -----------")
        for i in range(len(poprun_command) - 1):
            print("%s \\" % (poprun_command[i]))
        print("%s" % (poprun_command[len(poprun_command) - 1]))
        print("---------------------------------------")

        # Launch
        subprocess.run(" ".join(poprun_command), shell=True)


if __name__ == '__main__':
    ipu_launch = IPULaunch.parse_ipu_args()
    ipu_launch.launch()
