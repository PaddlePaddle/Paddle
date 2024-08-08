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

import copy
import os
import signal
import sys

from paddle.distributed.launch.job.container import Container
from paddle.distributed.launch.job.job import Job
from paddle.distributed.launch.job.pod import Pod

from .master import Master
from .watcher import Watcher


class ControllerMode:
    COLLECTIVE = "collective"
    PS = "ps"
    IPU = "ipu"
    RPC = "rpc"


class ControllerBase:
    def __init__(self, ctx):
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGABRT, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        if ctx.is_auto_tuner_mode():
            if not ctx.run_best:
                # set per task timeout
                signal.signal(signal.SIGALRM, self.not_exit_signal_handler)
                signal.alarm(ctx.max_time_per_task)
            else:
                signal.alarm(0)

        self.ctx = ctx
        self.master = Master.factory(self.ctx)

        self.watcher = Watcher(self.ctx)

        self.job = Job(
            nnodes=self.ctx.args.nnodes,
            mode=self.ctx.args.run_mode,
            jid=self.ctx.args.job_id,
        )
        self.pod = Pod()

        self.ctx.set_envs({"POD_NAME": self.pod.name})

        self.join_server = None

    def deploy_pod(self):
        assert (
            len(self.pod.containers) + len(self.pod.init_containers) > 0
        ), "No container in the pod"

        self.ctx.logger.info(f"Run {self.pod}")
        if len(self.pod.init_containers) > 0:
            self.ctx.logger.debug(self.pod.init_containers[0])
        if len(self.pod.containers) > 0:
            self.ctx.logger.debug(self.pod.containers[0])

        self.save_pod_env()
        self.ctx.status.run()
        self.pod.deploy()

    def run(self):
        self.build_job()
        self.build_pod()

        self.deploy_pod()

        self.watch()

    def watch(self) -> bool:
        '''
        watch self and peer status, return true to exit
        '''
        # TODO(kuizhiqing) unify ctx.status and master status

        self.ctx.logger.info(f"Watching {self.pod}")

        while not self.ctx.status.is_done():
            status = self.pod.watch(timeout=2)

            # if self.ctx.continuous_log():
            # default to print log
            self.pod.logs()

            # completed
            if status == self.ctx.status.COMPLETED:
                self.ctx.status.complete()

                self.master.set_status(status)

                while self.pod.logs():
                    pass

                self.ctx.logger.info(f"Pod {status}")
                return True

            # self failure
            elif status == self.ctx.status.FAILED:
                self.ctx.status.fail()

                self.master.set_status(status)
                self.master.restart_peer()

                fc = self.pod.failed_container()
                self.ctx.logger.info(f"Pod {status}")
                self.ctx.logger.error(f"Container failed !!!\n{fc[0]}")
                self.ctx.logger.info(
                    "------------------------- ERROR LOG DETAIL -------------------------"
                )
                fc[0].tail()

                if self.ctx.args.elastic_level <= 0:
                    self.pod.stop(timeout=3)
                    return True
                else:
                    self.pod.stop(timeout=30)
                    return False

            # peer failure
            if (
                self.ctx.status.is_restarting()
                and self.master.get_status() != self.ctx.status.COMPLETED
            ):
                # when peer failure, stop peer
                if self.ctx.args.elastic_level == -1:
                    self.pod.stop(timeout=3)
                    return True

                self.pod.stop(timeout=30)
                return False

    def stop(self, sigint=None):
        self.ctx.logger.debug("Controller stop")

        self.watcher.stop()

        self.master.stop()
        self.pod.stop(timeout=30)

    def finalize(self, exit=True):
        self.pod.join()
        self.master.stop()

        self.ctx.logger.info(f"Exit code {self.pod.exit_code}")
        if exit:
            sys.exit(self.pod.exit_code)

    def signal_handler(self, sigint, frame):
        if hasattr(self, 'sigint'):
            self.ctx.logger.info("Force quit in 10 seconds...")
            self.pod.stop(timeout=10)
            sys.exit(sigint)

        self.ctx.logger.info(f"Terminating with signal {sigint}")

        self.sigint = sigint
        self.ctx.status.done()
        self.stop(sigint=sigint)
        self.ctx.logger.info(f"Exit with signal {sigint}")
        sys.exit(sigint)

    def not_exit_signal_handler(self, sigint, frame):
        if hasattr(self, 'sigint'):
            self.ctx.logger.info("Force quit in 10 seconds...")
            self.pod.stop(timeout=10)

        self.ctx.logger.info(f"Terminating with signal {sigint}")

        self.sigint = sigint
        self.ctx.status.done()
        self.stop(sigint=sigint)
        self.ctx.logger.info(f"Exit with signal {sigint}")


class Controller(ControllerBase):
    '''
    Controller API for customization
    '''

    def build_job(self):
        '''
        build job fill the job info.
        '''
        self.ctx.logger.info(self.job)

    def build_pod(self) -> bool:
        '''
        build pod includes creating containers etc.

        Return True if succeed
        '''
        raise NotImplementedError

    def _get_entrypoint(self):
        if self.ctx.args.training_script.endswith('.py'):
            if os.environ.get("WITH_COVERAGE") == "ON":
                entrypoint = [
                    sys.executable,
                    "-u",
                    "-m",
                    "coverage",
                    "run",
                    "--branch",
                    "-p",
                    self.ctx.args.training_script,
                ]
            else:
                entrypoint = [
                    sys.executable,
                    "-u",
                    self.ctx.args.training_script,
                ]
        elif self.ctx.args.training_script.endswith('.pyxes'):
            entrypoint = [sys.executable, self.ctx.args.training_script]
        else:
            entrypoint = [self.ctx.args.training_script]

        entrypoint.extend(self.ctx.args.training_script_args)
        return entrypoint

    def _get_out_err_file(self, out=None, err=None):
        if out and self.ctx.args.log_dir != "":
            out = os.path.join(self.ctx.args.log_dir, out)
        if err and self.ctx.args.log_dir != "":
            err = os.path.join(self.ctx.args.log_dir, err)
        return out, (err or out)

    def new_container(
        self, entrypoint=None, envs={}, use_ctx_env=True, out=None, err=None
    ):
        c = Container(
            entrypoint=(entrypoint or self._get_entrypoint()),
            env=(self.ctx.get_envs() if use_ctx_env else {}),
            overwrite_log=self.ctx.args.log_overwrite,
        )
        c.outfile, c.errfile = self._get_out_err_file(out, err)
        c.update_env(envs)
        return c

    def add_container(
        self,
        container=None,
        entrypoint=None,
        envs={},
        log_file=None,
        is_init=False,
    ):
        if not container:
            envs = copy.deepcopy(envs)
            envs['PADDLE_LOG_DIR'] = str(os.path.abspath(self.ctx.args.log_dir))
            container = self.new_container(
                entrypoint=entrypoint, envs=envs, out=log_file, err=log_file
            )

        if is_init:
            self.pod.add_init_container(container)
        else:
            self.pod.add_container(container)

    def pod_replicas(self):
        '''
        how many process/container should be run in pod
        '''

        if self.ctx.args.nproc_per_node:
            return int(self.ctx.args.nproc_per_node)
        elif self.ctx.args.devices:
            return len(self.ctx.args.devices.split(','))
        else:
            return self.ctx.node.device.count

    def save_pod_log(self, info):
        '''
        save_pod_log append *info* to the log file of pod.name
        '''
        if not self.ctx.args.log_dir:
            return

        f = os.path.join(
            self.ctx.args.log_dir,
            f'{self.job.id}.{self.pod.name}.log',
        )
        try:
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'a+') as fd:
                if fd.tell() == 0:
                    fd.write(str(os.environ))
                    fd.write("\n")
                fd.write(str(info))
                fd.write("\n")
        except Exception as e:
            self.ctx.logger.error(f"save log failed because {e}")

    def save_pod_env(self):
        assert (
            len(self.pod.containers) + len(self.pod.init_containers) > 0
        ), "No container in the pod"

        if not self.ctx.args.log_dir:
            return

        for c in self.pod.init_containers:
            self._save_container_env(c, is_init=True)

        for c in self.pod.containers:
            self._save_container_env(c)

    def _save_container_env(self, container, is_init=False):
        f = os.path.join(
            self.ctx.args.log_dir,
            (
                f'envlog.init.{container.rank}'
                if is_init
                else f'envlog.{container.rank}'
            ),
        )
        try:
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, container.log_mode) as fd:
                for k, v in sorted(container.env.items()):
                    fd.write(str(f"{k}={v}\n"))
        except Exception as e:
            self.ctx.logger.error(f"save pod env log failed because {e}")
