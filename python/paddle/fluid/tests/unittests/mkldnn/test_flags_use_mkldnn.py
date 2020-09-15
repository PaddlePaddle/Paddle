# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import unicode_literals
from __future__ import print_function

import unittest
import os
import sys
import subprocess


class TestFlagsUseMkldnn(unittest.TestCase):
    def setUp(self):
        self._python_interp = sys.executable
        self._python_interp += " check_flags_use_mkldnn.py"

        self.env = os.environ.copy()
        self.env[str("GLOG_v")] = str("3")
        self.env[str("DNNL_VERBOSE")] = str("1")
        self.env[str("FLAGS_use_mkldnn")] = str("1")

    def test_flags_use_mkl_dnn(self):
        cmd = self._python_interp

        proc = subprocess.Popen(
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env)

        out, err = proc.communicate()
        returncode = proc.returncode

        print('out', out)
        print('err', err)

        assert returncode == 0
        # in python3, type(out) is 'bytes', need use encode
        assert out.find(
            "dnnl_verbose,exec,cpu,eltwise,jit:avx512_common,forward_training,"
            "data_f32::blocked:abc:f0 diff_undef::undef::f0,,alg:eltwise_relu".
            encode()) != -1


if __name__ == '__main__':
    unittest.main()
