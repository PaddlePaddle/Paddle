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
        print("default main prog:", main_prog)

        save_dir = "./run_save_model"
        fs = LocalFS()

        fs.delete(save_dir)
        print("begin _run_save_model")

        compiled, data_loader, optimizer, loss, image, label = self._init_env(
            exe, main_prog, startup_prog)
        for i in range(3):
            for data in data_loader():
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

        vars = list(
            filter(fluid.io.is_persistable, compiled._program.list_vars()))
        for var in vars:
            print("var:", var.name)

        m1 = PaddleModel(exe, compiled)
        m1.serialize(save_dir)

        m2 = PaddleModel(exe, compiled)
        m2.deserialize(save_dir)

        print("end _run_save_model")
        fs.delete(save_dir)

    # break at epoch 0: not save epoch

    def _run_save_0(self):
        fs = LocalFS()
        save_dir = "./run_save_0"
        fs.delete(save_dir)

        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog)

        print("main_prog:", main_prog)

        o = None
        i = 0
        name = None
        for i in acp.train_epoch_range(3, 0):
            o = acp._get_train_epoch_range()
            name = o.name
            print("name:", o.name, "epoch_no:", i)

            for data in data_loader():
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

            print("run_save_0 begin", i)
            vars = list(
                filter(fluid.io.is_persistable, compiled._program.list_vars()))
            for var in vars:
                print("var:", var.name)
            print("run_save_0 end", i)

            fluid.io.save_inference_model(
                save_dir, [image.name, label.name], [loss],
                exe,
                main_program=main_prog)
            assert len(o._exe_status) == 1, "there must be only 1 exestatus"

        o = acp._get_train_epoch_range()
        assert o == None, "now train epoch must not exits now"
        self.assertEqual(i, 2)
        fluid.io.save_inference_model(
            save_dir, [image.name, label.name], [loss],
            exe,
            main_program=compiled)

        fs.delete(save_dir)
        return name

    def _reset_generator(self):
        unique_name.generator = fluid.unique_name.UniqueNameGenerator()
        acp.generator = fluid.unique_name.UniqueNameGenerator()

    def _run_load_0(self):
        exe, main_prog, startup_prog = self._generate()

        fs = LocalFS()
        save_dir = "./run_save_0"
        fs.delete(save_dir)

        compiled, data_loader, optimizer, loss, image, label = self._init_env(
            exe, main_prog, startup_prog)

        o = None
        i = 0
        for i in acp.train_epoch_range(3, 0):
            o = acp._get_train_epoch_range()
            for data in data_loader():
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

            print("name:", o.name, "epoch_no:", i, "exe_status num:",
                  len(o._exe_status))
            fluid.io.save_inference_model(
                save_dir, [image.name, label.name], [loss],
                exe,
                main_program=main_prog)
            self.assertEqual(len(o._exe_status), 1)

        o = acp._get_train_epoch_range()
        assert o == None, "now train epoch must not exits now"
        self.assertEqual(i, 2)
        fluid.io.save_inference_model(
            save_dir, [image.name, label.name], [loss],
            exe,
            main_program=compiled)

        fs.delete(save_dir)

    # break at epoch 1: saved epoch_no is 0
    def _run_save_1(self, exe, data_loader, main_program):
        pass

    # break at epoch 9: saved epoch_no is 9
    def _run_save_9(self, exe, data_loader, main_program):
        pass

    # check two exe status
    def _run_save_2_exe(self, exe, data_loader, main_program):
        for i in acp.train_eoch_range(10):
            if i == 5:
                break

            name1 = acp._get_train_epoch_range().name
            for data in data_loader():
                fetch = exe.run(main_program, feed=data, fetch_list=[loss])
                print("fetch:", loss)

            for data in data_loader():
                fetch = exe.run(main_program, feed=data, fetch_list=[loss])
                print("fetch:", loss)

    def _run_save_multi_loop(self, exe, data_loader, main_program):
        for i in acp.train_eoch_range(10):
            name1 = acp._get_train_epoch_range().name
            for data in data_loader():
                fetch = exe.run(main_program, feed=data, fetch_list=[loss])
                print("fetch:", loss)

            for data in data_loader():
                fetch = exe.run(main_program, feed=data, fetch_list=[loss])
                print("fetch:", loss)

        assert acp._get_train_epoch_range() == None

        # range must has uniq name
        a = []
        for i in acp.train_eoch_range(10):
            name2 = acp._get_train_epoch_range().name
            a.append(i)
            for data in data_loader():
                fetch = exe.run(main_program, feed=data, fetch_list=[loss.name])
                print("fetch:", loss)
        assert acp._get_train_epoch_range() == None

        self.assertEqual(len(a), 10, "a must run from 0 to 9")
        self.assertNotEqual(name1, name2, "range must has uniq name")

        return name1, name2

    def _run_save(self, exe, data_loader, main_program):
        pass

    def _run_load(self, exe, main_program, name1, name2):
        pass

    def _generate(self):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        #main_prog = fluid.default_main_program()
        #startup_prog = fluid.default_startup_program()
        exe = fluid.Executor(places[0])

        return exe, main_prog, startup_prog

    def test_without_fleet(self):
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

    """
    def test_with_fleet(self):
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
