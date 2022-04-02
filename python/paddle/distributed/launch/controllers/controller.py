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

import sys
import os
import signal

from paddle.distributed.launch.job.job import Job
from paddle.distributed.launch.job.pod import Pod
from paddle.distributed.launch.job.container import Container

from .master import Master

import time


class ControleMode:
    COLLECTIVE = "collective"
    PS = "ps"


class ControllerBase(object):
    def __init__(self, ctx):
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGABRT, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        self.ctx = ctx
        self.master = Master.factory(self.ctx)

        self.job = Job(nnodes=self.ctx.args.nnodes,
                       mode=self.ctx.args.run_mode,
                       jid=self.ctx.args.job_id)
        self.pod = Pod()

        self.join_server = None

    def deploy_pod(self):

        assert len(self.pod.containers) > 0, "No container in the pod"

        self.ctx.logger.info("Run {}".format(self.pod))
        self.ctx.logger.debug(self.pod.containers[0])

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
        #TODO(kuizhiqing) unify ctx.status and master status

        self.ctx.logger.info("Watching {}".format(self.pod))

        while not self.ctx.status.is_done():
            status = self.pod.watch(timeout=2)

            if self.ctx.continous_log():
                self.pod.logs()

            # completed
            if status == self.ctx.status.COMPLETED:
                self.ctx.status.complete()

                self.master.set_status(status)

                self.ctx.logger.info("Pod {}".format(status))
                return True

            # self failure
            elif status == self.ctx.status.FAILED:
                self.ctx.status.fail()

                self.master.set_status(status)
                self.master.restart_peer()

                fc = self.pod.failed_container()
                self.ctx.logger.info("Pod {}".format(status))
                self.ctx.logger.error("Container failed !!!\n{}".format(fc[0]))
                fc[0].tail()
                self.pod.stop()

                if self.ctx.args.elastic_level <= 0:
                    return True
                else:
                    return False

            # peer failure
            if self.ctx.status.is_restarting() and self.master.get_status(
            ) != self.ctx.status.COMPLETED:
                self.pod.stop()
                return False

    def stop(self, sigint=None):
        self.ctx.logger.debug("Controller stop")
        self.master.stop()
        self.pod.stop(sigint)

    def finalize(self):
        self.pod.join()
        self.master.stop()

        self.ctx.logger.info("Exit code {}".format(self.pod.exit_code))
        sys.exit(self.pod.exit_code)

    def signal_handler(self, sigint, frame):
        self.ctx.logger.info("Terminating with signal {}".format(sigint))

        if hasattr(self, 'sigint'):
            time.sleep(5)
            sys.exit(sigint)

        self.sigint = sigint
        self.ctx.status.done()
        self.stop(sigint)
        time.sleep(1)
        self.ctx.logger.debug("Exit with signal {}".format(sigint))
        sys.exit(sigint)


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
        entrypoint = [sys.executable, "-u", self.ctx.args.training_script]
        entrypoint.extend(self.ctx.args.training_script_args)
        return entrypoint

    def _get_out_err_file(self, out=None, err=None):
        if out and self.ctx.args.log_dir != "":
            out = os.path.join(self.ctx.args.log_dir, out)
        if err and self.ctx.args.log_dir != "":
            err = os.path.join(self.ctx.args.log_dir, err)
        return out, (err or out)

    def new_container(self,
                      entrypoint=None,
                      envs={},
                      use_ctx_env=True,
                      out=None,
                      err=None):
        c = Container(
            entrypoint=(entrypoint or self._get_entrypoint()),
            env=(self.ctx.get_envs() if use_ctx_env else {}), )
        c.outfile, c.errfile = self._get_out_err_file(out, err)
        c.update_env(envs)
        return c

    def add_container(self,
                      container=None,
                      entrypoint=None,
                      envs={},
                      log_tag=None,
                      is_init=False):
        if not is_init and log_tag is not None:
            log_file = "{}.{}.{}.log".format(self.job.id, self.pod.name,
                                             log_tag)
        else:
            log_file = None

        if not container:
            container = self.new_container(
                entrypoint=entrypoint, envs=envs, out=log_file, err=log_file)

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

        f = os.path.join(self.ctx.args.log_dir,
                         '{}.{}.log'.format(self.job.id, self.pod.name))
        try:
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'a+') as fd:
                fd.write(str(info))
        except Exception as e:
            self.ctx.logger.error("save log failed because {}".format(e))
