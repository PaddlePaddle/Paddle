# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from op_test import is_custom_device

from paddle.base import core


def _read_flag_from_subprocess(flag_name: str, env: dict) -> str:
    code = f"""
import paddle
v = paddle.get_flags(\"{flag_name}\")
if isinstance(v, dict):
    v = v.get(\"{flag_name}\")
print(v)
"""
    return subprocess.check_output(
        [sys.executable, "-c", code],
        env=env,
        text=True,
    ).strip()


class TestFlagsEnv(unittest.TestCase):
    @unittest.skipIf(
        not (core.is_compiled_with_cuda() or is_custom_device()),
        "core is not compiled with CUDA",
    )
    def test_env_cudnn_deterministic_take_effect(self):
        env = os.environ.copy()
        env["FLAGS_cudnn_deterministic"] = "1"
        output = _read_flag_from_subprocess("FLAGS_cudnn_deterministic", env)
        self.assertIn(output, {"True", "1"})


if __name__ == "__main__":
    unittest.main()
