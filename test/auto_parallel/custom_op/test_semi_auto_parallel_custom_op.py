# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import collective.test_communication_api_base as test_base

from paddle.framework import core
from paddle.utils.cpp_extension.extension_utils import run_cmd


class TestCusomOp(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=200, nnode=1)
        self._default_envs = {"dtype": "float32", "seed": "2023"}
        self._changeable_envs = {"backend": ["cpu", "gpu"]}
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # compile, install the custom op egg into site-packages under background
        if os.name == 'nt':
            cmd = f'cd /d {cur_dir} && python custom_relu_setup.py install'
        else:
            cmd = (
                f'cd {cur_dir} && {sys.executable} custom_relu_setup.py install'
            )
        run_cmd(cmd)

    # test dynamic auto parallel run
    def test_dynamic_auto_parallel(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "semi_auto_parallel_for_custom_op.py",
                user_defined_envs=envs,
            )

    # test static spmd rule register
    def test_static_rule(self):
        import custom_relu  # noqa: F401 # pylint: disable=unused-import

        assert core.contains_spmd_rule("custom_relu")


if __name__ == "__main__":
    unittest.main()
