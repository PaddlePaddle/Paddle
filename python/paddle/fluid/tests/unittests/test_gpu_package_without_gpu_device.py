#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import subprocess
import unittest
<<<<<<< HEAD
import paddle
import tempfile
import paddle.fluid as fluid
=======
import tempfile
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
from paddle.fluid import core


class TestGPUPackagePaddle(unittest.TestCase):
<<<<<<< HEAD

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDwon(self):
        self.temp_dir.cleanup()

    def test_import_paddle(self):
        if core.is_compiled_with_cuda():
            if core.is_compiled_with_rocm():
                os.environ['HIP_VISIBLE_DEVICES'] = ''
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
<<<<<<< HEAD
            test_file = os.path.join(self.temp_dir.name,
                                     'test_no_gpu_run_rand.py')
=======
            test_file = os.path.join(
                self.temp_dir.name, 'test_no_gpu_run_rand.py'
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            with open(test_file, 'w') as wb:
                cmd_test = """
import paddle
paddle.utils.run_check()
x = paddle.rand([3,4])
assert x.place.is_gpu_place() is False, "There is no CUDA device, but Tensor's place is CUDAPlace"
"""
                wb.write(cmd_test)

            _python = sys.executable

            ps_cmd = '{} {}'.format(_python, test_file)
<<<<<<< HEAD
            ps_proc = subprocess.Popen(ps_cmd.strip().split(" "),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       env=os.environ)
=======
            ps_proc = subprocess.Popen(
                ps_cmd.strip().split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ,
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            stdout, stderr = ps_proc.communicate()

            assert 'CPU device will be used by default' in str(
                stderr
            ), "GPU version Paddle is installed. But CPU device can't be used when CUDA device is not set properly"
            assert "AssertionError" not in str(
                stderr
            ), "There is no CUDA device, but Tensor's place is CUDAPlace"


if __name__ == '__main__':
    unittest.main()
