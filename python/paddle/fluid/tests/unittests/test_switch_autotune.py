# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import unittest
import numpy


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = paddle.nn.Conv2D(1, 2, (3, 3))

    def forward(self, image, label=None):
        return self.conv(image)


def train_dygraph(net, data):
    out = net(data)
    loss = paddle.mean(out)
    adam = paddle.optimizer.Adam(parameters=net.parameters())
    out.backward()
    adam.step()
    adam.clear_grad()


def static_program(net, data):
    out = net(data)
    loss = paddle.mean(out)
    adam = paddle.optimizer.Adam()
    adam.minimize(loss)
    return loss


def set_flags(enable_autotune):
    if paddle.is_compiled_with_cuda():
        if enable_autotune:
            paddle.set_flags({'FLAGS_conv_workspace_size_limit': -1})
            paddle.set_flags({'FLAGS_cudnn_exhaustive_search': 1})
        else:
            paddle.set_flags({'FLAGS_conv_workspace_size_limit': 512})
            paddle.set_flags({'FLAGS_cudnn_exhaustive_search': 0})


class TestAutoTune(unittest.TestCase):
    def test_autotune(self):
        paddle.fluid.core.disable_autotune()
        status = paddle.fluid.core.autotune_status()
        self.assertEqual(status["use_autotune"], False)

        paddle.fluid.core.enable_autotune()
        status = paddle.fluid.core.autotune_status()
        self.assertEqual(status["use_autotune"], True)

    def check_status(self, expected_res):
        status = paddle.fluid.core.autotune_status()
        for key in status.keys():
            self.assertEqual(status[key], expected_res[key])


class TestDygraphAutoTuneStatus(TestAutoTune):
    def run_program(self, enable_autotune):
        set_flags(enable_autotune)
        if enable_autotune:
            paddle.fluid.core.enable_autotune()
        else:
            paddle.fluid.core.disable_autotune()
        paddle.fluid.core.autotune_range(1, 2)
        x_var = paddle.uniform((1, 1, 8, 8), dtype='float32', min=-1., max=1.)
        net = SimpleNet()
        for i in range(3):
            train_dygraph(net, x_var)
            if i >= 1 and i < 2:
                expected_res = {
                    "step_id": i,
                    "use_autotune": enable_autotune,
                    "cache_size": 0,
                    "cache_hit_rate": 0
                }
                self.check_status(expected_res)
            else:
                expected_res = {
                    "step_id": i,
                    "use_autotune": False,
                    "cache_size": 0,
                    "cache_hit_rate": 0
                }
                self.check_status(expected_res)

    def func_enable_autotune(self):
        self.run_program(enable_autotune=True)

    def test_enable_autotune(self):
        with paddle.fluid.framework._test_eager_guard():
            self.func_enable_autotune()
        self.func_enable_autotune()

    def func_disable_autotune(self):
        self.run_program(enable_autotune=False)

    def test_disable_autotune(self):
        with paddle.fluid.framework._test_eager_guard():
            self.func_disable_autotune()
        self.func_disable_autotune()


class TestStaticAutoTuneStatus(TestAutoTune):
    def run_program(self, enable_autotune):
        paddle.enable_static()
        set_flags(enable_autotune)
        if enable_autotune:
            paddle.fluid.core.enable_autotune()
        else:
            paddle.fluid.core.disable_autotune()
        paddle.fluid.core.autotune_range(1, 2)

        data_shape = [1, 1, 8, 8]
        data = paddle.static.data(name='X', shape=data_shape, dtype='float32')
        net = SimpleNet()
        loss = static_program(net, data)
        place = paddle.CUDAPlace(0) if paddle.fluid.core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        x = numpy.random.random(size=data_shape).astype('float32')

        for i in range(3):
            exe.run(feed={'X': x}, fetch_list=[loss])
            status = paddle.fluid.core.autotune_status()
            # In static mode, the startup_program will run at first.
            # The expected step_id will be increased by 1.
            if i >= 0 and i < 1:
                expected_res = {
                    "step_id": i + 1,
                    "use_autotune": enable_autotune,
                    "cache_size": 0,
                    "cache_hit_rate": 0
                }
                self.check_status(expected_res)
            else:
                expected_res = {
                    "step_id": i + 1,
                    "use_autotune": False,
                    "cache_size": 0,
                    "cache_hit_rate": 0
                }
                self.check_status(expected_res)
        paddle.disable_static()

    def func_enable_autotune(self):
        self.run_program(enable_autotune=True)

    def test_enable_autotune(self):
        with paddle.fluid.framework._test_eager_guard():
            self.func_enable_autotune()
        self.func_enable_autotune()

    def func_disable_autotune(self):
        self.run_program(enable_autotune=False)

    def test_disable_autotune(self):
        with paddle.fluid.framework._test_eager_guard():
            self.func_disable_autotune()
        self.func_disable_autotune()


if __name__ == '__main__':
    unittest.main()
