# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import subprocess
import sys
import unittest


def run_test_with_env(env_name):
    cmd = '''import paddle
a = paddle.zeros([2, 2])
b = paddle.ones([2, 2])
c = a + b
assert c.place.is_cpu_place(), f"The expected place is CPU, but got {c.place}"
    '''

    env = os.environ.copy()
    env[env_name] = ""
    return subprocess.run([sys.executable, "-c", cmd], env=env)


class TestEmptyVisibleDevices(unittest.TestCase):
    def test_xpu_env(self):
        ret = run_test_with_env("XPU_VISIBLE_DEVICES")
        self.assertEqual(ret.returncode, 0)

    def test_cuda_env(self):
        ret = run_test_with_env("CUDA_VISIBLE_DEVICES")
        self.assertEqual(ret.returncode, 0)


if __name__ == "__main__":
    unittest.main()
