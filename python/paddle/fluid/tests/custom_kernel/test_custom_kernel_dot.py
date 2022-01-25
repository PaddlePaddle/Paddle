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
import sys
import site
import subprocess
import unittest
import numpy as np


# use dot <CPU, ANY, INT8> as test case.
class TestCustomKernelDot1(unittest.TestCase):
    # invalid path check
    def test_dot_run_form_error_dir(self):
        INVALID_PATH = '/1/2/invalid_path'
        os.environ['CUSTOM_DEVICE_ROOT'] = INVALID_PATH
        err_msg = 'Error: Invalid CUSTOM_DEVICE_ROOT:[' + INVALID_PATH + '], path not exist. Please check.'
        try:
            import paddle
        except Exception as e:
            self.assertTrue(str(e).strip() == err_msg)


class TestCustomKernelDot2(unittest.TestCase):
    # user set path check
    def test_dot_run_form_cur_dir(self):
        # set environment for loading and registering compiled custom kernels
        os.environ['CUSTOM_DEVICE_ROOT'] = os.path.dirname(
            os.path.abspath(__file__))
        # test dot run
        x_data = np.random.uniform(1, 5, [2, 10]).astype(np.int8)
        y_data = np.random.uniform(1, 5, [2, 10]).astype(np.int8)
        result = np.sum(x_data * y_data, axis=1).reshape([2, 1])

        import paddle
        paddle.set_device('cpu')
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        out = paddle.dot(x, y)

        self.assertTrue(
            np.array_equal(out.numpy(), result),
            "custom kernel dot out: {},\n numpy dot out: {}".format(out.numpy(),
                                                                    result))


class TestCustomKernelDot3(unittest.TestCase):
    # defalut path check
    def test_dot_run_form_default_dir(self):
        # get paddle site
        paddle_lib_path = ''
        site_dirs = site.getsitepackages() if hasattr(
            site, 'getsitepackages'
        ) else [x for x in sys.path if 'site-packages' in x]
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
        default_path = os.path.sep.join(
            [paddle_lib_path, '..', '..', 'paddle-plugins'])
        # move so to defalut path
        cmd = 'mkdir -p {} && cp ./*.so {}'.format(default_path, default_path)
        subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)
        # test dot run
        x_data = np.random.uniform(1, 5, [2, 10]).astype(np.int8)
        y_data = np.random.uniform(1, 5, [2, 10]).astype(np.int8)
        result = np.sum(x_data * y_data, axis=1).reshape([2, 1])

        import paddle
        paddle.set_device('cpu')
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        out = paddle.dot(x, y)

        self.assertTrue(
            np.array_equal(out.numpy(), result),
            "custom kernel dot out: {},\n numpy dot out: {}".format(out.numpy(),
                                                                    result))
        cmd = 'rm -rf {}'.format(default_path)
        subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)


if __name__ == '__main__':
    # compile so and set to current path
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if os.name == 'nt':
        exit()
    else:
        # --inplace to place output so file to current dir
        cmd = 'cd {} && {} custom_kernel_dot_setup.py build_ext --inplace'.format(
            cur_dir, sys.executable)
        subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)
    unittest.main()
