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
        self._python_interp += " check_flags_use_mkldnn.py"

        self.env = os.environ.copy()
        self.env["GLOG_v"] = "1"
        self.env["DNNL_VERBOSE"] = "1"
        self.env["FLAGS_use_mkldnn"] = "1"

        self.relu_regex = b"^onednn_verbose,primitive,exec,cpu,eltwise,.+alg:eltwise_relu alpha:0 beta:0,10x20x30"

    def _print_when_false(self, cond, out, err):
        if not cond:
            print('out', out)
            print('err', err)
        return cond

    def found(self, regex, out, err):
        _found = re.search(regex, out, re.MULTILINE)
        return self._print_when_false(_found, out, err)

    def test_flags_use_mkl_dnn(self):
        cmd = self._python_interp

        proc = subprocess.Popen(
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
        )

        out, err = proc.communicate()
        returncode = proc.returncode

        assert returncode == 0
        assert self.found(self.relu_regex, out, err)


if __name__ == '__main__':
    unittest.main()
