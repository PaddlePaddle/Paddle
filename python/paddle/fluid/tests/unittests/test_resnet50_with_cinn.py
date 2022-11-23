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

import logging
import numpy as np
import paddle
import unittest

paddle.enable_static()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
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
class TestResnet50Accuracy(unittest.TestCase):

    def reader(self, limit):
        for _ in range(limit):
            yield {'image': np.random.randint(0, 256, size=[32, 3, 224, 224]).astype('float32'), \
                   'label': np.random.randint(0, 1000, size=[32]).astype('int64')}

    def generate_random_data(self, loop_num=10):
        feed = []
        data = self.reader(loop_num)
        for _ in range(loop_num):
            feed.append(next(data))
        return feed

    def build_program(self, main_program, startup_program):
        with paddle.static.program_guard(main_program, startup_program):
            image = paddle.static.data(name='image',
                                       shape=[32, 3, 224, 224],
                                       dtype='float32')
            label = paddle.static.data(name='label', shape=[32], dtype='int64')

            # TODO: stop_gradient slower training speed, need fix
            image.stop_gradient = False

            model = paddle.vision.models.resnet50()
            prediction = model(image)

            loss = paddle.nn.functional.cross_entropy(input=prediction,
                                                      label=label)
            loss = paddle.mean(loss)
            adam = paddle.optimizer.Adam(learning_rate=0.001)
            adam.minimize(loss)
        return loss

    def train(self, place, iters, feed, use_cinn=False, seed=1234):
        np.random.seed(seed)
        paddle.seed(seed)
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        set_cinn_flag(use_cinn)

        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()

        loss = self.build_program(main_program, startup_program)
        exe = paddle.static.Executor(place)

        compiled_prog = paddle.static.CompiledProgram(
            main_program).with_data_parallel(loss_name=loss.name)
        loss_vals = []
        scope = paddle.static.Scope()

        with paddle.static.scope_guard(scope):
            exe.run(startup_program)
            for step in range(iters):
                loss_v = exe.run(compiled_prog,
                                 feed=feed[step],
                                 fetch_list=[loss],
                                 return_numpy=True)
                loss_vals.append(loss_v[0][0])
        return loss_vals

    def test_check_resnet50_accuracy(self):
        place = paddle.CUDAPlace(
            0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()

        loop_num = 10
        feed = self.generate_random_data(loop_num)

        loss_c = self.train(place, loop_num, feed, use_cinn=True)
        loss_p = self.train(place, loop_num, feed, use_cinn=False)
        print("Losses of CINN:")
        print(loss_c)
        print("Losses of Paddle")
        print(loss_p)
        np.testing.assert_allclose(loss_c, loss_p, rtol=1e-05, atol=1e-05)


if __name__ == '__main__':
    unittest.main()
