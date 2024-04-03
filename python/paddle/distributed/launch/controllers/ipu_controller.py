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

import argparse
import os
import sys

from paddle.distributed.launch.job.container import Container

from .collective import CollectiveController, ControllerMode


class IPUController(CollectiveController):
    @classmethod
    def enable(cls, ctx):
        if ctx.args.training_script == "ipu":
            ctx.logger.debug(f"{cls.__name__} enabled")
            ctx.args.run_mode = ControllerMode.IPU
            return True
        else:
            return False

    def parse_ipu_args(self, args_list):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--hosts", type=str, help="The hosts for IPU distributed training."
        )
        parser.add_argument(
            "--nproc_per_host",
            type=int,
            help="The number of processes launched per host.",
        )
        parser.add_argument(
            "--ipus_per_replica",
            type=int,
            help="The number of IPUs requested per replica.",
        )
        parser.add_argument(
            "--ipu_partition",
            type=str,
            help="The partition name of IPU devices.",
        )
        parser.add_argument(
            "--vipu_server", type=str, help="The ip of the IPU device manager."
        )
        parser.add_argument(
            "training_script",
            type=str,
            help="The full path to the IPU distributed training program/script to be launched in parallel. e.g., ``training.py``.",
        )
        parser.add_argument('training_script_args', nargs=argparse.REMAINDER)
        return parser.parse_args(args_list)

    def replace_training_script(self):
        # IPU distributed computing is based on PopRun which is a wrapper of MPI.
        self.ctx.args.training_script = "poprun"
        poprun_args = self.parse_ipu_args(self.ctx.args.training_script_args)

        num_ipus = int(self.ctx.args.devices)
        # The number of replicas for data parallel
        assert (
            num_ipus % poprun_args.ipus_per_replica
        ) == 0, f"The number of IPUs:{num_ipus} mod the number of IPUs per replica:{poprun_args.ipus_per_replica} must == 0"
        num_replicas = num_ipus // poprun_args.ipus_per_replica
        self.ctx.logger.info(f"The number of total replicas is {num_replicas}.")

        # The number of processes
        num_nodes = len(poprun_args.hosts.split(','))
        num_procs = num_nodes * poprun_args.nproc_per_host
        self.ctx.logger.info(f"The number of total processes is {num_procs}.")
        assert (
            num_replicas % num_procs
        ) == 0, f"The number of replicas:{num_replicas} mod the number of processes:{num_procs} must == 0"

        # hosts and endpoints
        hosts = poprun_args.hosts.replace(' ', '').split(',')
        endpoints = [x + ":8090" for x in hosts]

        # args for poprun
        poprun_command = []

        poprun_command.append(f'--num-instances={num_procs}')
        poprun_command.append(f'--num-replicas={num_replicas}')
        poprun_command.append(
            f'--ipus-per-replica={poprun_args.ipus_per_replica}'
        )
        poprun_command.append('--host={}'.format(','.join(hosts)))
        poprun_command.append(f'--vipu-partition={poprun_args.ipu_partition}')
        poprun_command.append(f'--vipu-server-host={poprun_args.vipu_server}')

        poprun_command.extend(
            [
                '--update-partition=no',
                '--vipu-server-timeout=120',
                '--print-topology=yes',
                '--numa-aware=yes',
            ]
        )

        # global envs
        global_envs = '--mpi-local-args=\''
        log_level = os.getenv('POPART_LOG_LEVEL', None)
        if log_level:
            global_envs += f'-x POPART_LOG_LEVEL={log_level} '
        global_envs += (
            '-x PADDLE_TRAINERS_NUM={} -x PADDLE_TRAINER_ENDPOINTS={}'.format(
                num_procs, ','.join(endpoints)
            )
        )
        global_envs += '\''
        poprun_command.append(global_envs)

        # local envs
        for idx in range(num_procs):
            cur_endpoint = endpoints[idx // poprun_args.nproc_per_host]
            rank_in_node = idx % poprun_args.nproc_per_host
            poprun_command.append(
                f'--instance-mpi-local-args={idx}:\"-x PADDLE_TRAINER_ID={idx} -x PADDLE_CURRENT_ENDPOINT={cur_endpoint} -x PADDLE_RANK_IN_NODE={rank_in_node}\"'
            )

        # executor
        poprun_command.append(sys.executable)

        # script and script args
        poprun_command.append(poprun_args.training_script)
        poprun_command.extend(poprun_args.training_script_args)

        # for debug
        print("-----------  PopRun Command -----------")
        print("poprun \\")
        for i in range(len(poprun_command) - 1):
            print("%s \\" % (poprun_command[i]))
        print("%s" % (poprun_command[len(poprun_command) - 1]))
        print("---------------------------------------")

        # replace training_script_args
        self.ctx.args.training_script_args = poprun_command

    def _get_entrypoint(self):
        entrypoint = [self.ctx.args.training_script]
        entrypoint.extend(self.ctx.args.training_script_args)
        entrypoint = [" ".join(entrypoint)]
        return entrypoint

    def new_container(
        self, entrypoint=None, envs={}, use_ctx_env=True, out=None, err=None
    ):
        c = Container(
            entrypoint=(entrypoint or self._get_entrypoint()),
            env=(self.ctx.get_envs() if use_ctx_env else {}),
        )
        c.outfile, c.errfile = self._get_out_err_file(out, err)
        c.update_env(envs)
        # Need subprocess.Popen(shell=True) for PopRun command
        c.shell = True
        return c

    def run(self):
        # Replace the training script with the PopRun command
        self.replace_training_script()

        self.build_job()
        self.build_pod()

        self.deploy_pod()

        self.watch()
