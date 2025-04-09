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

import importlib
import os
import shlex
import site
import sys
import unittest

import numpy as np

import paddle

MODULE_NAME = "custom_raw_op_kernel_op_lib"


def prepare_module_path():
    # NOTE(Aurelius84): Normally, it's no need to add following codes for users.
    # But we simulate to pip install in current process, so interpreter don't snap
    # sys.path has been updated. So we update it manually.

    # See: https://stackoverflow.com/questions/56974185/import-runtime-installed-module-using-pip-in-python-3
    if os.name == 'nt':
        # NOTE(zhouwei25): getsitepackages on windows will return a list: [python install dir, site packages dir]
        site_dir = site.getsitepackages()[1]
    else:
        site_dir = site.getsitepackages()[0]
    custom_egg_path = [x for x in os.listdir(site_dir) if MODULE_NAME in x]
    assert (
        len(custom_egg_path) == 1
    ), f"Matched egg number is {len(custom_egg_path)}."
    sys.path.append(os.path.join(site_dir, custom_egg_path[0]))


# FIXME(zengjinle): do not know how to get the _compile_dir argument
# on Windows CI when compiling the custom op. Skip it on Windows CI
# temporarily.
@unittest.skipIf(os.name == "nt", "Windows does not support yet.")
class TestCustomRawReluOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, "custom_raw_op_kernel_op_setup.py")
        cmd = [sys.executable, path, "install", "--force"]
        if os.name != 'nt':
            install_lib = f"--install-lib={site.getsitepackages()[0]}"
            cmd.append(install_lib)
        cmd = " ".join([shlex.quote(c) for c in cmd])
        os.environ['MODULE_NAME'] = MODULE_NAME
        assert os.system(cmd) == 0
        prepare_module_path()

    @classmethod
    def tearDownClass(cls):
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y", MODULE_NAME]
        cmd = " ".join([shlex.quote(c) for c in cmd])
        assert os.system(cmd) == 0

    def custom_raw_relu(self, x):
        module = importlib.import_module(MODULE_NAME)
        custom_raw_relu_op = module.custom_raw_relu
        self.assertIsNotNone(custom_raw_relu_op)
        return custom_raw_relu_op(x)

    def test_static(self):
        paddle.enable_static()
        shape = [2, 3]
        x = paddle.static.data(name="x", dtype='float32', shape=shape)
        y1 = self.custom_raw_relu(x)
        y2 = paddle.nn.ReLU()(x)

        exe = paddle.static.Executor()
        exe.run(paddle.static.default_startup_program())
        x_np = np.random.uniform(low=-1.0, high=1.0, size=[2, 3]).astype(
            'float32'
        )
        y1_value, y2_value = exe.run(
            paddle.static.default_main_program(),
            feed={x.name: x_np},
            fetch_list=[y1, y2],
        )
        np.testing.assert_array_equal(y1_value, y2_value)

        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
