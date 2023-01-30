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

import os
<<<<<<< HEAD
import site
import sys
import unittest

=======
import sys
import site
import unittest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import numpy as np


class TestCustomKernelLoad(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        # compile so and set to current path
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        # --inplace to place output so file to current dir
<<<<<<< HEAD
        cmd = (
            'cd {} && {} custom_kernel_dot_setup.py build_ext --inplace'.format(
                cur_dir, sys.executable
            )
        )
=======
        cmd = 'cd {} && {} custom_kernel_dot_setup.py build_ext --inplace'.format(
            cur_dir, sys.executable)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        os.system(cmd)

        # get paddle lib path and place so
        paddle_lib_path = ''
<<<<<<< HEAD
        site_dirs = (
            site.getsitepackages()
            if hasattr(site, 'getsitepackages')
            else [x for x in sys.path if 'site-packages' in x]
        )
=======
        site_dirs = site.getsitepackages() if hasattr(
            site, 'getsitepackages') else [
                x for x in sys.path if 'site-packages' in x
            ]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        for site_dir in site_dirs:
            lib_dir = os.path.sep.join([site_dir, 'paddle', 'libs'])
            if os.path.exists(lib_dir):
                paddle_lib_path = lib_dir
                break
        if paddle_lib_path == '':
            if hasattr(site, 'USER_SITE'):
                lib_dir = os.path.sep.join([site.USER_SITE, 'paddle', 'libs'])
                if os.path.exists(lib_dir):
                    paddle_lib_path = lib_dir
        self.default_path = os.path.sep.join(
<<<<<<< HEAD
            [paddle_lib_path, '..', '..', 'paddle-plugins']
        )
        # copy so to default path
        cmd = 'mkdir -p {} && cp ./*.so {}'.format(
            self.default_path, self.default_path
        )
=======
            [paddle_lib_path, '..', '..', 'paddle-plugins'])
        # copy so to default path
        cmd = 'mkdir -p {} && cp ./*.so {}'.format(self.default_path,
                                                   self.default_path)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        os.system(cmd)  # wait

    def test_custom_kernel_dot_load(self):
        # test dot load
        x_data = np.random.uniform(1, 5, [2, 10]).astype(np.int8)
        y_data = np.random.uniform(1, 5, [2, 10]).astype(np.int8)
        result = np.sum(x_data * y_data, axis=1).reshape([2, 1])

        import paddle
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.set_device('cpu')
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        out = paddle.dot(x, y)

        np.testing.assert_array_equal(
            out.numpy(),
            result,
            err_msg='custom kernel dot out: {},\n numpy dot out: {}'.format(
<<<<<<< HEAD
                out.numpy(), result
            ),
        )
=======
                out.numpy(), result))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def tearDown(self):
        cmd = 'rm -rf {}'.format(self.default_path)
        os.system(cmd)


if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        exit()
    unittest.main()
