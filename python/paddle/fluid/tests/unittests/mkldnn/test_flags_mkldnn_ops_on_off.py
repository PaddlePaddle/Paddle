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
        self._python_interp += " check_flags_mkldnn_ops_on_off.py"

        self.env = os.environ.copy()
        self.env[str("DNNL_VERBOSE")] = str("1")
        self.env[str("FLAGS_use_mkldnn")] = str("1")

        self.relu_str = "dnnl_verbose,exec,cpu,eltwise,jit:avx512_common,forward_training," \
                        "data_f32::blocked:abc:f0 diff_undef::undef::f0,,alg:eltwise_relu alpha:0 beta:0,10x20x20"
        self.ew_add_str = "dnnl_verbose,exec,cpu,binary,jit:uni,undef,src_f32::blocked:abc:f0 " \
                          "src_f32::blocked:abc:f0 dst_f32::blocked:abc:f0,,alg:binary_add,10x20x30:10x20x30 10x20x30"
        self.matmul_str = "dnnl_verbose,exec,cpu,matmul,gemm:jit,undef,src_f32::blocked:abc:f0 " \
                          "wei_f32::blocked:acb:f0 dst_f32::blocked:abc:f0,,,b10m20n20k30"

    def flags_use_mkl_dnn_common(self, e):
        cmd = self._python_interp
        env = dict(self.env, **e)
        proc = subprocess.Popen(
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env)

        out, err = proc.communicate()
        returncode = proc.returncode

        print('out', out)
        print('err', err)

        assert returncode == 0
        return out

    def found(self, res):
        return res != -1

    def test_flags_use_mkl_dnn_on_empty_off_empty(self):
        # in python3, type(out) is 'bytes', need use encode
        out = self.flags_use_mkl_dnn_common({})
        assert self.found(out.find(self.relu_str.encode()))
        assert self.found(out.find(self.ew_add_str.encode()))
        assert self.found(out.find(self.matmul_str.encode()))

    def test_flags_use_mkl_dnn_on(self):
        # in python3, type(out) is 'bytes', need use encode
        env = {str("FLAGS_tracer_mkldnn_ops_on"): str("relu")}
        out = self.flags_use_mkl_dnn_common(env)
        assert self.found(out.find(self.relu_str.encode()))
        assert not self.found(out.find(self.ew_add_str.encode()))
        assert not self.found(out.find(self.matmul_str.encode()))

    def test_flags_use_mkl_dnn_off(self):
        # in python3, type(out) is 'bytes', need use encode
        env = {str("FLAGS_tracer_mkldnn_ops_off"): str("matmul")}
        out = self.flags_use_mkl_dnn_common(env)
        assert self.found(out.find(self.relu_str.encode()))
        assert self.found(out.find(self.ew_add_str.encode()))
        assert not self.found(out.find(self.matmul_str.encode()))

    def test_flags_use_mkl_dnn_on_off(self):
        # in python3, type(out) is 'bytes', need use encode
        env = {
            str("FLAGS_tracer_mkldnn_ops_on"): str("elementwise_add"),
            str("FLAGS_tracer_mkldnn_ops_off"): str("matmul")
        }
        out = self.flags_use_mkl_dnn_common(env)
        assert not self.found(out.find(self.relu_str.encode()))
        assert self.found(out.find(self.ew_add_str.encode()))
        assert not self.found(out.find(self.matmul_str.encode()))


if __name__ == '__main__':
    unittest.main()
