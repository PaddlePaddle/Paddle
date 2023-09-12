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

import os
import unittest

import numpy as np

import paddle
import paddle.base.incubate.checkpoint.auto_checkpoint as acp
from paddle import base
from paddle.base import unique_name
from paddle.base.framework import program_guard

BATCH_NUM = 4
BATCH_SIZE = 1

# IMAGE_SIZE = 128
CLASS_NUM = 2

USE_GPU = False  # whether use GPU to run model
places = base.cuda_places() if USE_GPU else base.cpu_places()

logger = None


def get_logger():
    global logger
    logger = acp._get_logger(20)
    return logger


def get_random_images_and_labels(image_shape, label_shape):
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return image, label


def sample_list_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM):
            sample_list = []
            for _ in range(BATCH_SIZE):
                image, label = get_random_images_and_labels([4, 4], [1])
                sample_list.append([image, label])

            yield sample_list

    return __reader__


class AutoCheckpointBase(unittest.TestCase):
    def _init_env(
        self, exe, main_prog, startup_prog, minimize=True, iterable=True
    ):
        def simple_net():
            image = paddle.static.data(
                name='image', shape=[-1, 4, 4], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[-1, 1], dtype='int64'
            )

            fc_tmp = paddle.static.nn.fc(image, size=CLASS_NUM)
            cross_entropy = paddle.nn.functional.softmax_with_cross_entropy(
                fc_tmp, label
            )
            loss = paddle.mean(cross_entropy)
            sgd = paddle.optimizer.SGD(learning_rate=1e-3)
            if minimize:
                sgd.minimize(loss)
            return sgd, loss, image, label

        with program_guard(main_prog, startup_prog):
            sgd, loss, image, label = simple_net()

            if minimize:
                compiled = base.CompiledProgram(main_prog)
            else:
                compiled = None
            loader = base.io.DataLoader.from_generator(
                feed_list=[image, label],
                capacity=64,
                use_double_buffer=True,
                iterable=iterable,
            )

            loader.set_sample_list_generator(
                sample_list_generator_creator(), places[0]
            )

        if minimize:
            exe.run(startup_prog)

        return compiled, loader, sgd, loss, image, label

    def _generate(self):
        main_prog = base.Program()
        startup_prog = base.Program()
        exe = base.Executor(places[0])

        return exe, main_prog, startup_prog

    def _reset_generator(self):
        unique_name.generator = base.unique_name.UniqueNameGenerator()
        acp.generator = base.unique_name.UniqueNameGenerator()
        acp.g_acp_type = None
        acp.g_checker = acp.AutoCheckpointChecker()
        acp.g_program_attr = {}

    def _clear_envs(self):
        os.environ.pop("PADDLE_RUNNING_ENV", None)

    def _readd_envs(self):
        os.environ["PADDLE_RUNNING_ENV"] = "PADDLE_EDL_AUTO_CHECKPOINT"
