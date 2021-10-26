# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import logging
import numpy as np
import paddle
import unittest

paddle.enable_static()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def set_cinn_flag(val):
    cinn_compiled = False
    try:
        paddle.set_flags({'FLAGS_use_cinn': val})
        cinn_compiled = True
    except ValueError:
        logger.warning("The used paddle is not compiled with CINN.")
    return cinn_compiled


@unittest.skipIf(not set_cinn_flag(True), "Paddle is not compiled with CINN.")
class TestParallelExecutorRunCinn(unittest.TestCase):
    def test_run_from_cinn(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data = paddle.static.data(
                name='X', shape=[None, 1], dtype='float32')
            prediction = paddle.static.nn.fc(data, 2)
            loss = paddle.mean(prediction)
            adam = paddle.optimizer.Adam()
            adam.minimize(loss)

        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(startup_program)
        compiled_program = paddle.static.CompiledProgram(
            main_program).with_data_parallel(loss_name=loss.name)

        batch_size = 16
        x = np.random.random(size=(batch_size, 1)).astype('float32')
        fetch = exe.run(compiled_program,
                        feed={'X': x},
                        fetch_list=[prediction.name],
                        return_merged=False)

        set_cinn_flag(False)


@unittest.skipIf(not set_cinn_flag(True), "Paddle is not compiled with CINN.")
class TestAddReluAccuracy(unittest.TestCase):
    def reader(self, limit):
        for i in range(limit):
            yield np.random.random([1, 28]).astype('float32'), \
                np.random.random([1, 28]).astype('float32'), \
                np.random.randint(0, 2, size=[1]).astype('int64')

    def RandFeedData(self, loop_num=10):
        feed = []
        data = self.reader(loop_num)
        for i in range(loop_num):
            x, y, z = next(data)
            feed.append({'x': x, 'y': y, 'z': z})
        return feed

    def BuildProgram(self, main_program, startup_program):
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name='x', shape=[1, 28], dtype='float32')
            y = paddle.static.data(name="y", shape=[1, 28], dtype='float32')
            z = paddle.static.data(name="z", shape=[1], dtype='int64')

            hidden = paddle.add(x, y)
            prediction = paddle.nn.functional.relu(hidden)

            loss = paddle.nn.functional.cross_entropy(input=prediction, label=z)
            loss = paddle.mean(loss)
            sgd = paddle.optimizer.SGD(learning_rate=0.001)
            sgd.minimize(loss)
        return x, y, z, loss

    def Run(self, place, iters, feed, use_cinn=False):
        set_cinn_flag(use_cinn)

        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()

        x, y, z, loss = self.BuildProgram(main_program, startup_program)
        exe = paddle.static.Executor(place)

        parallel_exec = paddle.static.CompiledProgram(
            main_program).with_data_parallel(loss_name=loss.name)
        loss_vals = []
        scope = paddle.static.Scope()

        with paddle.static.scope_guard(scope):
            exe.run(startup_program)
            for step in range(iters):
                loss_v = exe.run(parallel_exec,
                                 feed=feed[step],
                                 fetch_list=[loss],
                                 return_numpy=True)
                loss_vals.append(loss_v[0][0])
        return loss_vals

    def test_check_addrelu_accuracy(self):
        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()

        loop_num = 10
        feed = self.RandFeedData(loop_num)

        loss_t = self.Run(place, loop_num, feed, use_cinn=True)
        loss_f = self.Run(place, loop_num, feed, use_cinn=False)

        max_err = np.max(np.fabs(np.asarray(loss_t) - np.asarray(loss_f)))
        self.assertLessEqual(max_err, 1e-5)


@unittest.skipIf(not set_cinn_flag(True), "Paddle is not compiled with CINN.")
class TestResnet50Accuracy(unittest.TestCase):
    def reader(self, limit):
        for i in range(limit):
            yield np.random.randint(0, 256, size=[1, 3, 224, 224]).astype('float32'), \
                  np.random.randint(0, 1000, size=[1]).astype('int64')

    def RandFeedData(self, loop_num=10):
        feed = []
        data = self.reader(loop_num)
        for i in range(loop_num):
            x, y = next(data)
            feed.append({'image': x, 'label': y})
        return feed

    def BuildProgram(self, main_program, startup_program):
        with paddle.static.program_guard(main_program, startup_program):
            image = paddle.static.data(
                name='image', shape=[1, 3, 224, 224], dtype='float32')
            label = paddle.static.data(name='label', shape=[1], dtype='int64')

            model = paddle.vision.models.resnet50()
            prediction = model(image)

            loss = paddle.nn.functional.cross_entropy(
                input=prediction, label=label)
            loss = paddle.mean(loss)
            adam = paddle.optimizer.Adam(learning_rate=0.001)
            adam.minimize(loss)
        return image, label, loss

    def Run(self, place, iters, feed, use_cinn=False):
        set_cinn_flag(use_cinn)

        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()

        x, y, loss = self.BuildProgram(main_program, startup_program)
        exe = paddle.static.Executor(place)

        parallel_exec = paddle.static.CompiledProgram(
            main_program).with_data_parallel(loss_name=loss.name)
        loss_vals = []
        scope = paddle.static.Scope()

        with paddle.static.scope_guard(scope):
            exe.run(startup_program)
            for step in range(iters):
                loss_v = exe.run(parallel_exec,
                                 feed=feed[step],
                                 fetch_list=[loss],
                                 return_numpy=True)
                loss_vals.append(loss_v[0][0])
        return loss_vals

    def test_check_resnet50_accuracy(self):
        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()

        loop_num = 10
        feed = self.RandFeedData(loop_num)

        loss_t = self.Run(place, loop_num, feed, use_cinn=True)
        loss_f = self.Run(place, loop_num, feed, use_cinn=False)

        max_err = np.max(np.fabs(np.asarray(loss_t) - np.asarray(loss_f)))
        self.assertLessEqual(max_err, 1e-5)


if __name__ == '__main__':
    unittest.main()
