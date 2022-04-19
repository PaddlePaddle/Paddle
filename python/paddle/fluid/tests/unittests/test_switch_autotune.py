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
import numpy as np


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = paddle.nn.Conv2D(1, 2, (3, 3))

    def forward(self, image, label=None):
        return self.conv(image)


def train_dygraph(net, data):
    data.stop_gradient = False
    out = net(data)
    loss = paddle.mean(out)
    adam = paddle.optimizer.Adam(parameters=net.parameters())
    out.backward()
    adam.step()
    adam.clear_grad()


def static_program(net, data):
    data.stop_gradient = False
    out = net(data)
    loss = paddle.mean(out)
    adam = paddle.optimizer.Adam()
    adam.minimize(loss)
    return loss


class TestAutoTune(unittest.TestCase):
    def set_flags(self, enable_autotune):
        if paddle.is_compiled_with_cuda():
            if enable_autotune:
                paddle.set_flags({'FLAGS_conv_workspace_size_limit': -1})
            else:
                paddle.set_flags({'FLAGS_conv_workspace_size_limit': 512})

    def get_flags(self, name):
        res = paddle.get_flags(name)
        return res[name]

    def get_expected_res(self, step_id, enable_autotune):
        expected_res = {
            "step_id": step_id,
            "cache_size": 0,
            "cache_hit_rate": 0
        }
        if paddle.is_compiled_with_cuda():
            # Total 3 * num_iters cache accesses, only iter 2 hits the cache.
            if enable_autotune and step_id >= 1:
                expected_res["cache_size"] = 3
            if enable_autotune and step_id == 2:
                expected_res["cache_hit_rate"] = np.round(
                    float(3) / float(9), 5)
        return expected_res

    def test_autotune(self):
        paddle.fluid.core.disable_autotune()
        self.assertEqual(self.get_flags("FLAGS_use_autotune"), False)

        paddle.fluid.core.enable_autotune()
        self.assertEqual(self.get_flags("FLAGS_use_autotune"), True)

    def check_status(self, expected_res):
        status = paddle.fluid.core.autotune_status()
        for key in status.keys():
            if key == "cache_hit_rate":
                v = np.round(status[key], 5)
            else:
                v = status[key]
            self.assertEqual(v, expected_res[key])


class TestDygraphAutoTuneStatus(TestAutoTune):
    def run_program(self, enable_autotune):
        self.set_flags(enable_autotune)
        if enable_autotune:
            paddle.fluid.core.enable_autotune()
        else:
            paddle.fluid.core.disable_autotune()
        paddle.fluid.core.set_autotune_range(1, 2)
        x_var = paddle.uniform((1, 1, 8, 8), dtype='float32', min=-1., max=1.)
        net = SimpleNet()
        for i in range(3):
            train_dygraph(net, x_var)
            expected_res = self.get_expected_res(i, enable_autotune)
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

        data_shape = [1, 1, 8, 8]
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data = paddle.static.data(
                name='X', shape=data_shape, dtype='float32')
            net = SimpleNet()
            loss = static_program(net, data)
        place = paddle.CUDAPlace(0) if paddle.fluid.core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(startup_program)
        x = np.random.random(size=data_shape).astype('float32')

        self.set_flags(enable_autotune)
        if enable_autotune:
            paddle.fluid.core.enable_autotune()
        else:
            paddle.fluid.core.disable_autotune()
        paddle.fluid.core.set_autotune_range(1, 2)

        for i in range(3):
            exe.run(program=main_program, feed={'X': x}, fetch_list=[loss])
            status = paddle.fluid.core.autotune_status()
            expected_res = self.get_expected_res(i, enable_autotune)
            self.check_status(expected_res)
        paddle.disable_static()

    def func_enable_autotune(self):
        self.run_program(enable_autotune=True)

    def test_enable_autotune(self):
        self.func_enable_autotune()

    def func_disable_autotune(self):
        self.run_program(enable_autotune=False)

    def test_disable_autotune(self):
        self.func_disable_autotune()


if __name__ == '__main__':
    unittest.main()
