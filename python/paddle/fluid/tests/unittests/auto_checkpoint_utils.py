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


def get_logger():
    global logger
    logger = acp._get_logger(20)
    return logger


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


class AutoCheckpointBase(unittest.TestCase):
    def setUp(self):
        get_logger()
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

    def _init_env(self, exe, main_prog, startup_prog, minimize=True):
        def simple_net():
            image = fluid.data(
                name='image', shape=[-1, 16, 16], dtype='float32')
            label = fluid.data(name='label', shape=[-1, 1], dtype='int64')

            fc_tmp = fluid.layers.fc(image, size=CLASS_NUM)
            cross_entropy = fluid.layers.softmax_with_cross_entropy(fc_tmp,
                                                                    label)
            loss = fluid.layers.reduce_mean(cross_entropy)
            sgd = fluid.optimizer.SGD(learning_rate=1e-3)
            if minimize:
                sgd.minimize(loss)
            return sgd, loss, image, label

        with program_guard(main_prog, startup_prog):
            sgd, loss, image, label = simple_net()

            if minimize:
                compiled = fluid.CompiledProgram(main_prog).with_data_parallel(
                    loss_name=loss.name)
            else:
                compiled = None

            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            loader = DataLoader(
                dataset,
                feed_list=[image, label],
                places=places,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=2)

        if minimize:
            exe.run(startup_prog)

        return compiled, loader, sgd, loss, image, label

    def _generate(self):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        exe = fluid.Executor(places[0])

        return exe, main_prog, startup_prog

    def _reset_generator(self):
        unique_name.generator = fluid.unique_name.UniqueNameGenerator()
        acp.generator = fluid.unique_name.UniqueNameGenerator()
