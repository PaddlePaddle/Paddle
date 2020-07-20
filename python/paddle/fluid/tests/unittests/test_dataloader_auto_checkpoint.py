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

from paddle.fluid.incubate.fleet.utils.fs import LocalFS
from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient
import paddle.fluid.incubate.checkpoint.auto_checkpoint as acp
import paddle.fluid.incubate.checkpoint.dataloader_auto_checkpoint as dacp
import paddle.fluid.incubate.checkpoint.auto_checkpoint as acp
from paddle.fluid.incubate.checkpoint.checkpoint_saver import PaddleModel
from paddle.fluid.framework import program_guard
from paddle.fluid import unique_name

import numpy as np
from paddle.io import Dataset, BatchSampler, DataLoader

from paddle.fluid.tests.unittests.auto_checkpoint_utils import AutoCheckpointBase, get_logger
logger = get_logger()


class DataLoaderAutoCheckpointTest(AutoCheckpointBase):
    def _run_must_acp(self):
        """
        check acp_type muast acp.
        """
        logger.info("enter _run_acp")
        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog, iterable=True)

        # use two
        for i in acp.train_epoch_range(2):
            for data in data_loader():
                name = acp.g_train_epoch_range.name
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

        self.assertEqual(acp.g_acp_type, acp.CONST_ACP_TYPE)
        self.assertEqual(acp.g_train_epoch_range, None)
        self.assertEqual(i, 1)

        # use two
        for i in range(2):
            for data in data_loader():
                self.assertEqual(acp.g_acp_type, acp.CONST_ACP_TYPE)

                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

        self.assertEqual(acp.g_acp_type, acp.CONST_ACP_TYPE)
        self.assertEqual(acp.g_train_epoch_range, None)
        self.assertEqual(i, 1)
        logger.info("exit _run_acp")

    def _run_must_dacp(self):
        """
        check acp_type muast dacp.
        """
        logger.info("enter _run_must_dacp")
        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog, iterable=True)

        # use two
        for i in range(2):
            for data in data_loader():
                name = acp.g_train_epoch_range.name
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

        self.assertEqual(acp.g_acp_type, acp.CONST_DACP_TYPE)
        self.assertEqual(acp.g_train_epoch_range, None)
        self.assertEqual(i, 1)

        # use two
        for i in acp.train_epoch_range(2):
            self.assertEqual(acp.g_acp_type, acp.CONST_DACP_TYPE)
            self.assertEqual(acp.g_train_epoch_range, None)
            for data in data_loader():
                name = acp.g_train_epoch_range.name
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

        self.assertEqual(acp.g_acp_type, acp.CONST_DACP_TYPE)
        self.assertEqual(acp.g_train_epoch_range, None)
        self.assertEqual(i, 1)
        logger.info("exit _run_must_dacp")

    def _run_save_basic(self, break_epoch_no=None):
        """
        save checkpoint
        """
        logger.info("enter _run_save_basic")
        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog)

        i = 0
        name = None
        logger.info("g_acp_type:{} g_ranges:{}".format(acp.g_acp_type,
                                                       dacp.g_ranges))
        for i in range(3):
            for data in data_loader():
                name = data_loader._auto_checkpoint_name
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

            if break_epoch_no is not None:
                if i == break_epoch_no:
                    break

        self.assertEqual(acp.g_acp_type, acp.CONST_DACP_TYPE)
        self.assertEqual(len(dacp.g_ranges), 1, "There must be one element")

        if break_epoch_no is None:
            self.assertEqual(i, 2)
        else:
            self.assertEqual(i, break_epoch_no)
        logger.info("leave _run_save_basic")

    def _run_load_basic(self, break_epoch_no=None):
        """
        load checkpoint
        """
        logger.info("enter _run_load_basic")
        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog)

        i = 0
        name = None
        epochs = []
        for i in range(3):
            for data in data_loader():
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])
                if i not in epochs:
                    epochs.append(i)

        self.assertEqual(len(dacp.g_ranges), 1, "There must be one element")
        if break_epoch_no is not None:
            if break_epoch_no == 0:
                self.assertEqual(epochs, [0, 1, 2])
            elif break_epoch_no == 1:
                self.assertEqual(epochs, [1, 2])
            elif break_epoch_no == 2:
                self.assertEqual(epochs, [2])
        else:
            self.assertEqual(epochs, [2])

        logger.info("leave _run_load_basic")

    def test_basic_type(self):
        checker = acp._get_checker()
        fs = HDFSClient(checker.hdfs_home, None)

        # test type must be right
        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()
        self._run_must_acp()
        self._reset_generator()
        self._run_must_dacp()

        fs.delete(checker.hdfs_checkpoint_path)

    def test_basic(self):
        checker = acp._get_checker()
        fs = HDFSClient(checker.hdfs_home, None)

        # test save and load epoch_no must be right
        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()
        self._run_save_basic()
        self._reset_generator()
        self._run_load_basic()

        fs.delete(checker.hdfs_checkpoint_path)

    def test_corener_epochno(self):
        checker = acp._get_checker()
        fs = HDFSClient(checker.hdfs_home, None)

        # test break at some epoch_nos
        for i in range(3):
            fs.delete(checker.hdfs_checkpoint_path)
            self._reset_generator()
            self._run_save_basic(break_epoch_no=i)
            self._reset_generator()
            self._run_load_basic(break_epoch_no=i)

        fs.delete(checker.hdfs_checkpoint_path)


if __name__ == '__main__':
    unittest.main()
