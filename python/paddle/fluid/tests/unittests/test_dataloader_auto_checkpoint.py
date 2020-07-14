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

    def _run_try_catch(self, break_epoch_no=None):
        logger.info("enter _run_try_catch")
        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog, iterable=False)

        # use two
        i = 0
        name = None
        for i in range(3):
            try:
                data_loader.start()
                while True:
                    name = acp.g_train_epoch_range.name
                    fetch = exe.run(compiled, fetch_list=[loss])

                if break_epoch_no is not None:
                    if i == break_epoch_no:
                        break
            except fluid.core.EOFException:
                logger.info("complete one epoch")
            finally:
                data_loader.reset()

        self.assertTrue(acp.g_acp_type, None)
        self.assertTrue(
            len(dacp.g_train_epoch_ranges), 1, "There must be one element")
        self.assertTrue(dacp.g_train_epoch_ranges[name], None,
                        "Running must be None")
        logger.info("exit _run_try_catch")

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

    """
    def _run_load_try_catch(self):
        exe, main_prog, startup_prog = self._generate()

        fs = LocalFS()
        save_dir = "./run_load_0"
        fs.delete(save_dir)

        compiled, data_loader, optimizer, loss, image, label = self._init_env(
            exe, main_prog, startup_prog)

        o = None
        i = 0
        check = False
        for i in acp.train_epoch_range(3, 0):
            print("_run_load_0 name:", o.name, "epoch_no:", i)
            if started_epoch_no is not None and not check:
                self.assertEqual(o.get(), started_epoch_no)
                check = True

            for data in data_loader():
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

            fluid.io.save_inference_model(
                save_dir, [image.name, label.name], [loss],
                exe,
                main_program=main_prog)
            self.assertEqual(len(o._exe_status), 1)

        o = acp._get_train_epoch_range()
        self.assertTrue(o == None, "now train epoch must not exits now")
        self.assertEqual(i, 2)
        fluid.io.save_inference_model(
            save_dir, [image.name, label.name], [loss],
            exe,
            main_program=compiled)

        fs.delete(save_dir)
    """

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
        self._reset_generator()
        self._run_try_catch()

        fs.delete(checker.hdfs_checkpoint_path)

    def test_coreno_epoch(self):
        pass

    def test_distributed_basic(self):
        pass

    def test_multiple(self):
        pass


if __name__ == '__main__':
    unittest.main()
