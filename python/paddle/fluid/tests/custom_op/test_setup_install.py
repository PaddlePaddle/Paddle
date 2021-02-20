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
import unittest
import paddle
import subprocess
import numpy as np
from paddle.utils.cpp_extension.extension_utils import run_cmd
from paddle.utils.cpp_extension.extension_utils import use_new_custom_op_load_method

# switch to old custom op method
use_new_custom_op_load_method(False)


class TestSetUpInstall(unittest.TestCase):
    def setUp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # compile, install the custom op egg into site-packages under background
        cmd = 'cd {} && python setup_install.py install'.format(cur_dir)
        run_cmd(cmd)

        # NOTE(Aurelius84): Normally, it's no need to add following codes for users.
        # But we simulate to pip install in current process, so interpreter don't snap
        # sys.path has been updated. So we update it manually.

        # See: https://stackoverflow.com/questions/56974185/import-runtime-installed-module-using-pip-in-python-3
        site_dir = site.getsitepackages()[0]
        custom_egg_path = [
            x for x in os.listdir(site_dir) if 'custom_relu2' in x
        ]
        assert len(custom_egg_path) == 1, "Matched egg number is %d." % len(
            custom_egg_path)
        sys.path.append(os.path.join(site_dir, custom_egg_path[0]))

    def test_api(self):
        # usage: import the package directly
        import custom_relu2

        raw_data = np.array([[-1, 1, 0], [1, -1, -1]]).astype('float32')
        gt_data = np.array([[0, 1, 0], [1, 0, 0]]).astype('float32')
        x = paddle.to_tensor(raw_data, dtype='float32')
        # use custom api
        out = custom_relu2.relu2(x)
        out3 = custom_relu2.relu3(x)

        self.assertTrue(np.array_equal(out.numpy(), gt_data))
        self.assertTrue(np.array_equal(out3.numpy(), gt_data))


if __name__ == '__main__':
    unittest.main()
