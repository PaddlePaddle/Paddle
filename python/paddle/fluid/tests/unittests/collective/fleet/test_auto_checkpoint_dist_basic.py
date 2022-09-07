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

import unittest
import paddle
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.collective import CollectiveOptimizer, fleet
import os
import sys

from paddle.distributed.fleet.utils.fs import LocalFS, HDFSClient
import paddle.fluid.incubate.checkpoint.auto_checkpoint as acp
from paddle.fluid.incubate.checkpoint.checkpoint_saver import PaddleModel
from paddle.fluid.framework import program_guard
from paddle.fluid import unique_name

import numpy as np
from paddle.io import Dataset, BatchSampler, DataLoader

from paddle.fluid.tests.unittests.auto_checkpoint_utils import AutoCheckpointBase, get_logger
from test_auto_checkpoint import AutoCheckPointACLBase

paddle.enable_static()
logger = get_logger()


class AutoCheckpointTestDist(AutoCheckPointACLBase):

    def setUp(self):
        get_logger()
        logger.info("enter tests")

        self._old_environ = dict(os.environ)
        proc_env = {
            "PADDLE_RUNNING_ENV": "PADDLE_EDL_AUTO_CHECKPOINT",
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_RUNNING_PLATFORM": "PADDLE_CLOUD",
            "PADDLE_JOB_ID": "test_job_auto_dist_basic",
            "PADDLE_EDL_HDFS_HOME": "/usr/local/hadoop-2.7.7",
            "PADDLE_EDL_HDFS_NAME": "",
            "PADDLE_EDL_HDFS_UGI": "",
            "PADDLE_EDL_HDFS_CHECKPOINT_PATH": "auto_checkpoint_dist_basic",
            "PADDLE_EDL_ONLY_FOR_CE_TEST": "1",
            "PADDLE_EDL_FS_CACHE": ".auto_checkpoint_test_dist_basic",
            "PADDLE_EDL_SAVE_CHECKPOINT_INTER": "0"
        }
        os.environ.update(proc_env)

    def test_distributed_basic(self):
        checker = acp._get_checker()
        fs = HDFSClient(checker.hdfs_home, None)
        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()

        logger.info("begin test_distributed_basic")
        fs = LocalFS()
        save_dir = "./run_save_0"
        fs.delete(save_dir)

        # basic
        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog, minimize=False)

        # fleet
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:6070"

        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        with fluid.program_guard(main_prog, startup_prog):
            dist_optimizer = fleet.distributed_optimizer(optimizer)
            dist_optimizer.minimize(loss)

        exe.run(startup_prog)

        o = None
        i = 0
        name = None
        for i in acp.train_epoch_range(3, 0):
            o = acp._get_train_epoch_range()
            name = o.name
            logger.info("_run_save_0 name:{} epoch_no:{}".format(o.name, i))

            for data in data_loader():
                fetch = exe.run(fleet.main_program,
                                feed=data,
                                fetch_list=[loss])

            self.assertEqual(len(o._exe_status), 1)

        o = acp._get_train_epoch_range()
        assert o == None, "now train epoch must not exits now"
        self.assertEqual(i, 2)

        fs.delete(save_dir)

        logger.info("end test_distributed_basic")


if __name__ == '__main__':
    unittest.main()
