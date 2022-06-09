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

import subprocess
import sys
import pickle
import os
import unittest
import paddle


class TestDistTRT(unittest.TestCase):

    def setUp(self):
        self.init_case()
        self.script = "test_trt_c_allreduce_infer_script.py"

    def init_case(self):
        self.op_type = "c_allreduce_sum"

    def test_run(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        cmd = f"python -u -m paddle.distributed.fleet.launch --gpus 0,1 {self.script} {self.op_type}"
        cmd = cmd.split(" ")

        local_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env)

        local_out, local_err = local_proc.communicate()


class TestMin(TestDistTRT):

    def init_case(self):
        self.op_type = "c_allreduce_min"


class TestMax(TestDistTRT):

    def init_case(self):
        self.op_type = "c_allreduce_max"


class TestProd(TestDistTRT):

    def init_case(self):
        self.op_type = "c_allreduce_prod"


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
