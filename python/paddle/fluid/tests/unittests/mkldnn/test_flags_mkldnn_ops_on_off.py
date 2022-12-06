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

import os
import re
import subprocess
import sys
import unittest


class TestFlagsUseMkldnn(unittest.TestCase):
    def setUp(self):
        self._python_interp = sys.executable
        self._python_interp += " check_flags_mkldnn_ops_on_off.py"

        self.env = os.environ.copy()
        self.env[str("DNNL_VERBOSE")] = str("1")
        self.env[str("FLAGS_use_mkldnn")] = str("1")

        self.relu_regex = b"^onednn_verbose,exec,cpu,eltwise,.+alg:eltwise_relu alpha:0 beta:0,10x20x20"
        self.ew_add_regex = (
            b"^onednn_verbose,exec,cpu,binary.+alg:binary_add,10x20x30:10x20x30"
        )
        self.matmul_regex = (
            b"^onednn_verbose,exec,cpu,matmul,.*10x20x30:10x30x20:10x20x20"
        )

    def flags_use_mkl_dnn_common(self, e):
        cmd = self._python_interp
        env = dict(self.env, **e)
        proc = subprocess.Popen(
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        out, err = proc.communicate()
        returncode = proc.returncode

        assert returncode == 0
        return out, err

    def _print_when_false(self, cond, out, err):
        if not cond:
            print('out', out)
            print('err', err)
        return cond

    def found(self, regex, out, err):
        _found = re.search(regex, out, re.MULTILINE)
        return self._print_when_false(_found, out, err)

    def not_found(self, regex, out, err):
        _not_found = not re.search(regex, out, re.MULTILINE)
        return self._print_when_false(_not_found, out, err)

    def test_flags_use_mkl_dnn_on_empty_off_empty(self):
        out, err = self.flags_use_mkl_dnn_common({})
        assert self.found(self.relu_regex, out, err)
        assert self.found(self.ew_add_regex, out, err)
        assert self.found(self.matmul_regex, out, err)

    def test_flags_use_mkl_dnn_on(self):
        env = {str("FLAGS_tracer_mkldnn_ops_on"): str("relu")}
        out, err = self.flags_use_mkl_dnn_common(env)
        assert self.found(self.relu_regex, out, err)
        assert self.not_found(self.ew_add_regex, out, err)
        assert self.not_found(self.matmul_regex, out, err)

    def test_flags_use_mkl_dnn_on_multiple(self):
        env = {str("FLAGS_tracer_mkldnn_ops_on"): str("relu,elementwise_add")}
        out, err = self.flags_use_mkl_dnn_common(env)
        assert self.found(self.relu_regex, out, err)
        assert self.found(self.ew_add_regex, out, err)
        assert self.not_found(self.matmul_regex, out, err)

    def test_flags_use_mkl_dnn_off(self):
        env = {str("FLAGS_tracer_mkldnn_ops_off"): str("matmul_v2")}
        out, err = self.flags_use_mkl_dnn_common(env)
        assert self.found(self.relu_regex, out, err)
        assert self.found(self.ew_add_regex, out, err)
        assert self.not_found(self.matmul_regex, out, err)

    def test_flags_use_mkl_dnn_off_multiple(self):
        env = {str("FLAGS_tracer_mkldnn_ops_off"): str("matmul_v2,relu")}
        out, err = self.flags_use_mkl_dnn_common(env)
        assert self.not_found(self.relu_regex, out, err)
        assert self.found(self.ew_add_regex, out, err)
        assert self.not_found(self.matmul_regex, out, err)

    def test_flags_use_mkl_dnn_on_off(self):
        env = {
            str("FLAGS_tracer_mkldnn_ops_on"): str("elementwise_add"),
            str("FLAGS_tracer_mkldnn_ops_off"): str("matmul_v2"),
        }
        out, err = self.flags_use_mkl_dnn_common(env)
        assert self.not_found(self.relu_regex, out, err)
        assert self.found(self.ew_add_regex, out, err)
        assert self.not_found(self.matmul_regex, out, err)


if __name__ == '__main__':
    unittest.main()
