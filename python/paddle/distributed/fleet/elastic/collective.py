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

import os
import shutil
import tempfile

import paddle
from paddle.distributed.fleet.elastic.manager import LauncherInterface
from paddle.distributed.fleet.launch_utils import (
    logger,
    pull_worker_log,
    start_local_trainers,
)


class CollectiveLauncher(LauncherInterface):
    def __init__(self, args):
        self.args = args
        self.procs = []

    def launch(self):
        logger.info("collective launcher launch ...")
        args = self.args
        self.tmp_dir = tempfile.mkdtemp()
        cluster, pod = paddle.distributed.fleet.launch.get_cluster_info(args)
        global_envs = paddle.distributed.fleet.launch.get_global_envs(
            args, self.tmp_dir
        )

        self.procs = start_local_trainers(
            cluster,
            pod,
            training_script=args.training_script,
            training_script_args=args.training_script_args,
            log_dir=args.log_dir,
            envs=global_envs,
        )

        for idx, proc in enumerate(self.procs):
            logger.info(f"launch proc_id:{proc.proc.pid} idx:{idx}")

    def stop(self):
        logger.info("collective launcher stop ...")
        if not self._terminate_procs():
            logger.error("kill process failed")
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def watch(self):
        logger.debug("collective launcher watch ...")
        for p in self.procs:
            if p.log_fn and p.local_rank == 0:
                pull_worker_log(p)
        ret = self._check_procs()
        return ret
