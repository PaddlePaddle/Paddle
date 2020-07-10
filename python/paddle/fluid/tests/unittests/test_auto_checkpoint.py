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

BATCH_NUM = 20
BATCH_SIZE = 16

#IMAGE_SIZE = 128
CLASS_NUM = 10

USE_GPU = False  # whether use GPU to run model
places = fluid.cuda_places() if USE_GPU else fluid.cpu_places()

logger = None


# define a random dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([16, 16]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class AutoCheckpointTest(unittest.TestCase):
    def setUp(self):
        global logger
        logger = acp._get_logger(20)
        logger.info("enter tests")

        self._old_environ = dict(os.environ)
        proc_env = {
            "PADDLE_RUNNING_ENV": "PADDLE_EDL_AUTO_CHECKPOINT",
            "PADDLE_EDL_TRAINER_ID": "0",
            "PADDLE_RUNNING_PLATFORM": "PADDLE_CLOUD",
            "PADDLE_JOB_ID": "test_job1",
            "PADDLE_EDL_HDFS_HOME": "/usr/local/hadoop-2.7.7",
            "PADDLE_EDL_HDFS_NAME": "",
            "PADDLE_EDL_HDFS_UGI": "",
            "PADDLE_EDL_HDFS_CHECKPOINT_PATH": "checkpoint",
            "PADDLE_EDL_ONLY_FOR_CE_TEST": "1"
        }
        os.environ.update(proc_env)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_environ)

    def _init_env(self, exe, main_prog, startup_prog):
        def simple_net():
            image = fluid.data(
                name='image', shape=[-1, 16, 16], dtype='float32')
            label = fluid.data(name='label', shape=[-1, 1], dtype='int64')

            fc_tmp = fluid.layers.fc(image, size=CLASS_NUM)
            cross_entropy = fluid.layers.softmax_with_cross_entropy(fc_tmp,
                                                                    label)
            loss = fluid.layers.reduce_mean(cross_entropy)
            sgd = fluid.optimizer.SGD(learning_rate=1e-3)
            sgd.minimize(loss)
            return sgd, loss, image, label

        with program_guard(main_prog, startup_prog):
            sgd, loss, image, label = simple_net()

            compiled = fluid.CompiledProgram(main_prog).with_data_parallel(
                loss_name=loss.name)

            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            loader = DataLoader(
                dataset,
                feed_list=[image, label],
                places=places,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=2)

        exe.run(startup_prog)

        return compiled, loader, sgd, loss, image, label

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

    def _reset_generator(self):
        unique_name.generator = fluid.unique_name.UniqueNameGenerator()
        acp.generator = fluid.unique_name.UniqueNameGenerator()

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

    def _generate(self):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        exe = fluid.Executor(places[0])

        return exe, main_prog, startup_prog

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

    """
    def test_distributed_basic(self):
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:6070"

        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        exe, optimizer, avg_loss, main_program = self._init_model()

        dist_optimizer = fleet.distributed_optimizer(optimizer)
        dist_optimizer.minimize(avg_loss)

        self._run(exe, fleet.main_program)
    """


if __name__ == '__main__':
    unittest.main()
