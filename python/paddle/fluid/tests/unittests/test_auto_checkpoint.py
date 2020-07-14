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
from paddle.fluid.incubate.checkpointer.checkpointer import PaddleModel
from paddle.fluid.framework import program_guard
from paddle.fluid import unique_name

import numpy as np
from paddle.io import Dataset, BatchSampler, DataLoader

from paddle.fluid.tests.unittests.auto_checkpoint_utils import AutoCheckpointBase, get_logger

logger = get_logger()


class AutoCheckpointTest(AutoCheckpointBase):
    """
    def setUp(self):
        super(AutoCheckpointTest, self).setUp()

    def tearDown(self):
        super(AutoCheckpointTest, self).tearDown()
    """

    def _run_save_model(self):
        exe, main_prog, startup_prog = self._generate()
        #print("default main prog:", main_prog)

        save_dir = "./run_save_model"
        fs = LocalFS()

        fs.delete(save_dir)
        print("begin _run_save_model")

        compiled, data_loader, optimizer, loss, image, label = self._init_env(
            exe, main_prog, startup_prog)
        for i in range(3):
            for data in data_loader():
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

        m1 = PaddleModel(exe, compiled)
        m1.serialize(save_dir)

        m2 = PaddleModel(exe, compiled)
        m2.deserialize(save_dir)

        print("end _run_save_model")
        fs.delete(save_dir)

    def _run_save_0(self, break_epoch_no=None):
        fs = LocalFS()
        save_dir = "./run_save_0"
        fs.delete(save_dir)

        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog)

        o = None
        i = 0
        name = None
        for i in acp.train_epoch_range(3, 0):
            o = acp._get_train_epoch_range()
            name = o.name
            print("_run_save_0 name:", o.name, "epoch_no:", i)

            for data in data_loader():
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

            fluid.io.save_inference_model(
                save_dir, [image.name, label.name], [loss],
                exe,
                main_program=main_prog)
            self.assertEqual(len(o._exe_status), 1)

            if break_epoch_no is not None:
                if i == break_epoch_no:
                    break

        o = acp._get_train_epoch_range()
        assert o == None, "now train epoch must not exits now"
        if break_epoch_no is None:
            self.assertEqual(i, 2)
        else:
            self.assertEqual(i, break_epoch_no)

        fs.delete(save_dir)

    def _run_load_0(self, started_epoch_no=None):
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
            o = acp._get_train_epoch_range()

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

    def test_basic(self):
        logger.info("begin test_basic")
        checker = acp._get_checker()
        fs = HDFSClient(checker.hdfs_home, None)
        fs.delete(checker.hdfs_checkpoint_path)

        self._reset_generator()
        self._run_save_model()

        self._reset_generator()
        self._run_save_0()

        self._reset_generator()
        self._run_load_0()

        self._reset_generator()
        self._run_load_0()

        fs.delete(checker.hdfs_checkpoint_path)
        logger.info("end test_basic")

    def test_corner_epoch_no(self):
        logger.info("begin test_corener_epoch_no")
        checker = acp._get_checker()
        fs = HDFSClient(checker.hdfs_home, None)

        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()
        self._run_save_0(break_epoch_no=0)
        self._reset_generator()
        self._run_load_0(started_epoch_no=0)

        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()
        self._run_save_0(break_epoch_no=1)

        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()
        self._run_save_0(break_epoch_no=2)

        fs.delete(checker.hdfs_checkpoint_path)
        logger.info("end test_corener_epoch_no")

    def test_multiple(self):
        checker = acp._get_checker()
        fs = HDFSClient(checker.hdfs_home, None)
        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()

        logger.info("begin test_multiple")
        fs = LocalFS()
        save_dir = "./run_save_0"
        fs.delete(save_dir)

        exe, main_prog1, startup_prog1 = self._generate()
        _, main_prog2, startup_prog2 = self._generate()

        compiled1, data_loader1, optimizer1, loss1, image1, label1 = \
            self._init_env(exe, main_prog1, startup_prog1)

        compiled2, data_loader2, optimizer2, loss2, image2, label2 = \
            self._init_env(exe, main_prog2, startup_prog2)

        o = None
        for i in acp.train_epoch_range(3, 0):
            for data in data_loader1():
                fetch = exe.run(compiled1, feed=data, fetch_list=[loss1])

            for data in data_loader2():
                fetch = exe.run(compiled2, feed=data, fetch_list=[loss2])

            o = acp._get_train_epoch_range()
            self.assertTrue(len(o._exe_status), 2)

        o = acp._get_train_epoch_range()
        self.assertTrue(o == None, "now train epoch must not exits now")
        self.assertEqual(i, 2)

        fs.delete(save_dir)
        logger.info("end test_multiple")

    def test_distributed_basic(self):
        checker = acp._get_checker()
        fs = HDFSClient(checker.hdfs_home, None)
        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()

        logger.info("begin test_distributed_basic")
        fs = LocalFS()
        save_dir = "./run_save_0"
        fs.delete(save_dir)

        #basic
        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog, minimize=False)

        #fleet
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
            print("_run_save_0 name:", o.name, "epoch_no:", i)

            for data in data_loader():
                fetch = exe.run(fleet.main_program,
                                feed=data,
                                fetch_list=[loss])

            fluid.io.save_inference_model(
                save_dir, [image.name, label.name], [loss],
                exe,
                main_program=main_prog)
            self.assertEqual(len(o._exe_status), 1)

        o = acp._get_train_epoch_range()
        assert o == None, "now train epoch must not exits now"
        self.assertEqual(i, 2)

        fs.delete(save_dir)

        logger.info("end test_distributed_basic")


if __name__ == '__main__':
    unittest.main()
