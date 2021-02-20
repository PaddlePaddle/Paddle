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

import os
import sys
import site
import subprocess
import unittest
import numpy as np

import paddle
from paddle.utils.cpp_extension.extension_utils import run_cmd


class TestMultiOutputDtypes(unittest.TestCase):
    def setUp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # compile, install the custom op egg into site-packages under background
        cmd = 'cd {} && python multi_out_setup.py install'.format(cur_dir)
        run_cmd(cmd)

        # NOTE(Aurelius84): Normally, it's no need to add following codes for users.
        # But we simulate to pip install in current process, so interpreter don't snap
        # sys.path has been updated. So we update it manually.

        # See: https://stackoverflow.com/questions/56974185/import-runtime-installed-module-using-pip-in-python-3
        site_dir = site.getsitepackages()[0]
        custom_egg_path = [
            x for x in os.listdir(site_dir) if 'multi_out_module_setup' in x
        ]
        assert len(custom_egg_path) == 1, "Matched egg number is %d." % len(
            custom_egg_path)
        sys.path.append(os.path.join(site_dir, custom_egg_path[0]))

        # usage: import the package directly
        import multi_out_module_setup
        self.custom_op = multi_out_module_setup.multi_out
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']

    def run_static(self, device, dtype):
        paddle.set_device(device)
        x_data = np.random.uniform(-1, 1, [4, 8]).astype(dtype)

        with paddle.static.scope_guard(paddle.static.Scope()):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(name='X', shape=[None, 8], dtype=dtype)
                outs = self.custom_op(x)

                exe = paddle.static.Executor()
                exe.run(paddle.static.default_startup_program())
                res = exe.run(paddle.static.default_main_program(),
                              feed={'X': x_data},
                              fetch_list=outs)

                return res

    def check_multi_outputs(self, outs, is_dynamic=False):
        out, zero_float64, one_int32 = outs
        if is_dynamic:
            zero_float64 = zero_float64.numpy()
            one_int32 = one_int32.numpy()
        # Fake_float64
        self.assertTrue('float64' in str(zero_float64.dtype))
        self.assertTrue(
            np.array_equal(zero_float64, np.zeros([4, 8]).astype('float64')))
        # ZFake_int32
        self.assertTrue('int32' in str(one_int32.dtype))
        self.assertTrue(
            np.array_equal(one_int32, np.ones([4, 8]).astype('int32')))

    def test_static(self):
        paddle.enable_static()
        for device in self.devices:
            for dtype in self.dtypes:
                res = self.run_static(device, dtype)
                self.check_multi_outputs(res)
        paddle.disable_static()

    def test_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                paddle.set_device(device)
                x_data = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                x = paddle.to_tensor(x_data)
                outs = self.custom_op(x)

                self.assertTrue(len(outs) == 3)
                self.check_multi_outputs(outs, True)


if __name__ == '__main__':
    unittest.main()
