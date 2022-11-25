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
from paddle.fluid.tests.unittests.test_auto_checkpoint import AutoCheckPointACLBase

paddle.enable_static()
logger = get_logger()


class AutoCheckpointTest3(AutoCheckPointACLBase):

    def setUp(self):
        get_logger()
        logger.info("enter tests")

        self._old_environ = dict(os.environ)
        proc_env = {
            "PADDLE_RUNNING_ENV": "PADDLE_EDL_AUTO_CHECKPOINT",
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_RUNNING_PLATFORM": "PADDLE_CLOUD",
            "PADDLE_JOB_ID": "test_job_auto_3",
            "PADDLE_EDL_HDFS_HOME": "/usr/local/hadoop-2.7.7",
            "PADDLE_EDL_HDFS_NAME": "",
            "PADDLE_EDL_HDFS_UGI": "",
            "PADDLE_EDL_HDFS_CHECKPOINT_PATH": "auto_checkpoint_3",
            "PADDLE_EDL_ONLY_FOR_CE_TEST": "1",
            "PADDLE_EDL_FS_CACHE": ".auto_checkpoint_test_3",
            "PADDLE_EDL_SAVE_CHECKPOINT_INTER": "0"
        }
        os.environ.update(proc_env)

    def test_corner_epoch_no(self):
        self._test_corner_epoch_no(2)


if __name__ == '__main__':
    unittest.main()
