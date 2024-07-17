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

import json
import os
import tempfile
import unittest
import warnings

import numpy as np

import paddle


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
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
            "cache_hit_rate": 0,
        }
        if paddle.is_compiled_with_cuda():
            # Total 3 * num_iters cache accesses, only iter 2 hits the cache.
            expected_res["cache_size"] = 3
            expected_res["cache_hit_rate"] = (step_id + 0.0) / (step_id + 1.0)
        return expected_res

    def test_autotune(self):
        paddle.incubate.autotune.set_config(
            config={"kernel": {"enable": False}}
        )
        self.assertEqual(self.get_flags("FLAGS_use_autotune"), False)

        paddle.incubate.autotune.set_config(config={"kernel": {"enable": True}})
        self.assertEqual(self.get_flags("FLAGS_use_autotune"), True)

    def check_status(self, expected_res):
        status = paddle.base.core.autotune_status()
        for key in status.keys():
            v = status[key]
            if key == "cache_hit_rate":
                np.testing.assert_allclose(v, expected_res[key])
            else:
                np.testing.assert_array_equal(v, expected_res[key])


class TestDygraphAutoTuneStatus(TestAutoTune):
    def run_program(self, enable_autotune):
        self.set_flags(enable_autotune)
        if enable_autotune:
            paddle.incubate.autotune.set_config(
                config={"kernel": {"enable": True, "tuning_range": [1, 2]}}
            )
        else:
            paddle.incubate.autotune.set_config(
                config={"kernel": {"enable": False}}
            )
        x_var = paddle.uniform((1, 1, 8, 8), dtype='float32', min=-1.0, max=1.0)
        net = SimpleNet()
        for i in range(3):
            train_dygraph(net, x_var)
            expected_res = self.get_expected_res(i, enable_autotune)
            self.check_status(expected_res)

    def test_enable_autotune(self):
        self.run_program(enable_autotune=True)

    def test_disable_autotune(self):
        self.run_program(enable_autotune=False)


class TestStaticAutoTuneStatus(TestAutoTune):
    def run_program(self, enable_autotune):
        with paddle.pir_utils.OldIrGuard():
            paddle.enable_static()

            data_shape = [1, 1, 8, 8]
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                data = paddle.static.data(
                    name='X', shape=data_shape, dtype='float32'
                )
                net = SimpleNet()
                loss = static_program(net, data)
            place = (
                paddle.CUDAPlace(0)
                if paddle.base.core.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            exe.run(startup_program)
            x = np.random.random(size=data_shape).astype('float32')

            # Node(tizheng): warmup run to make sure the following runs
            # are in the same thread. Necessary for CUDNNv8 tests
            exe.run(program=main_program, feed={'X': x}, fetch_list=[loss])

            self.set_flags(enable_autotune)
            if enable_autotune:
                config = {"kernel": {"enable": True, "tuning_range": [1, 2]}}
                tfile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
                json.dump(config, tfile)
                tfile.close()
                paddle.incubate.autotune.set_config(tfile.name)
                os.remove(tfile.name)
            else:
                paddle.incubate.autotune.set_config(
                    config={"kernel": {"enable": False, "tuning_range": [1, 2]}}
                )

            for i in range(3):
                exe.run(program=main_program, feed={'X': x}, fetch_list=[loss])
                status = paddle.base.core.autotune_status()
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


class TestAutoTuneAPI(unittest.TestCase):
    def test_set_config_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            config = {"kernel": {"enable": 1, "tuning_range": 1}}
            tfile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
            json.dump(config, tfile)
            tfile.close()
            paddle.incubate.autotune.set_config(tfile.name)
            os.remove(tfile.name)
            self.assertTrue(len(w) == 2)

    def test_set_config_attr(self):
        paddle.incubate.autotune.set_config(config=None)
        self.assertEqual(
            paddle.get_flags("FLAGS_use_autotune")["FLAGS_use_autotune"], True
        )


if __name__ == '__main__':
    unittest.main()
