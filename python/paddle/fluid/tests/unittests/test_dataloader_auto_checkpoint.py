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
from paddle.fluid.incubate.fleet.collective import CollectiveOptimizer, fleet, TrainStatus
import os
import sys

from paddle.fluid.incubate.fleet.utils.fs import LocalFS
from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient
import paddle.fluid.incubate.checkpointer.auto_checkpoint as acp
import paddle.fluid.incubate.checkpointer.dataloader_auto_checkpoint as dacp
import paddle.fluid.incubate.checkpointer.auto_checkpoint as acp
from paddle.fluid.incubate.checkpointer.checkpointer import PaddleModel
from paddle.fluid.framework import program_guard
from paddle.fluid import unique_name

import numpy as np
from paddle.io import Dataset, BatchSampler, DataLoader

from paddle.fluid.tests.unittests.auto_checkpoint_utils import AutoCheckpointBase, get_logger
logger = get_logger()


class DataLoaderAutoCheckpointTest(AutoCheckpointBase):
    def _run_complex(self):
        logger.info("enter _run_complex")
        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog, iterable=True)

        # use two
        for i in acp.train_epoch_range(3):
            for data in data_loader():
                name = acp.g_train_epoch_range.name
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

        self.assertTrue(acp.g_acp_type, acp.CONST_ACP_TYPE)
        self.assertTrue(i, 2)
        logger.info("exit _run_complex")

    def _run_save_basic(self, break_epoch_no=None):
        logger.info("enter _run_save_basic")
        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog)

        i = 0
        name = None
        for i in range(3):
            for data in data_loader():
                print("loader:", data_loader._auto_checkpoint_name)
                name = data_loader._auto_checkpoint_name
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

            if break_epoch_no is not None:
                if i == break_epoch_no:
                    break

        self.assertEqual(acp.g_acp_type, None)
        self.assertEqual(
            len(dacp.g_train_epoch_ranges), 1, "There must be one element")

        if break_epoch_no is None:
            self.assertEqual(i, 2)
        else:
            self.assertEqual(i, break_epoch_no)
        logger.info("leave _run_save_basic")

    def _run_load_basic(self):
        logger.info("enter _run_load_basic")
        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog)

        i = 0
        name = None
        for i in range(3):
            for data in data_loader():
                print("loader:", data_loader._auto_checkpoint_name)
                name = data_loader._auto_checkpoint_name
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

            if break_epoch_no is not None:
                if i == break_epoch_no:
                    break

        self.assertEqual(acp.g_acp_type, None)
        self.assertEqual(
            len(dacp.g_train_epoch_ranges), 1, "There must be one element")

        if break_epoch_no is None:
            self.assertEqual(i, 2)
        else:
            self.assertEqual(i, break_epoch_no)
        logger.info("leave _run_load_basic")

    def test_basic(self):
        checker = acp._get_checker()
        fs = HDFSClient(checker.hdfs_home, None)

        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()
        self._run_complex()

        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()
        self._run_save_basic()

        fs.delete(checker.hdfs_checkpoint_path)

    def test_coreno_epoch(self):
        checker = acp._get_checker()
        fs = HDFSClient(checker.hdfs_home, None)

        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()
        self._run_save_basic()

        self._reset_generator()
        self._run_load_basic()
        fs.delete(checker.hdfs_checkpoint_path)


if __name__ == '__main__':
    unittest.main()
