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
<<<<<<< HEAD
import sys
import pickle
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
import os
import unittest
import paddle


class TestDistTRT(unittest.TestCase):
<<<<<<< HEAD

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def setUp(self):
        self.init_case()
        self.script = "test_trt_c_allreduce_infer_script.py"

    def init_case(self):
        self.op_type = "c_allreduce_sum"
<<<<<<< HEAD
        self.target_value = 4.
=======
        self.target_value = 4.0
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
        self.precision = "fp16"

    def test_run(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        cmd = f"python -u -m paddle.distributed.fleet.launch --gpus 0,1 {self.script} {self.op_type} {self.precision}"
        cmd = cmd.split(" ")

        local_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env)

        local_out, local_err = local_proc.communicate()
        for line in local_out.decode("utf-8").split("\n"):
            results = line.split("=")
            if len(results) == 2 and results[0] == "c_allreduce_out":
                self.assertEqual(float(results[1]), self.target_value)


class TestMin(TestDistTRT):
<<<<<<< HEAD

    def init_case(self):
        self.op_type = "c_allreduce_min"
        self.target_value = 2.
        self.precision = "int8"


#class TestMax(TestDistTRT):
=======
    def init_case(self):
        self.op_type = "c_allreduce_min"
        self.target_value = 2.0
        self.precision = "int8"


# class TestMax(TestDistTRT):
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
#
#    def init_case(self):
#        self.op_type = "c_allreduce_max"
#        self.target_value = 2.
#        self.precision = "fp16"
#
#
<<<<<<< HEAD
#class TestProd(TestDistTRT):
=======
# class TestProd(TestDistTRT):
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
#
#    def init_case(self):
#        self.op_type = "c_allreduce_prod"
#        self.target_value = 2.
#        self.precision = "fp16"

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
