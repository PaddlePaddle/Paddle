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
import shutil
import unittest

import paddle
from paddle.jit.dy2static.export_subgraph import get_saving_dir


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = paddle.pow(x, 2)
        z = x + y
        z2 = paddle.matmul(y, z)
        out = paddle.nn.functional.relu(z * z2)
        out = paddle.mean(out)
        return out, z2


class TestSaveFwdBwdProg(unittest.TestCase):
    def setUp(self):
        self.net = paddle.jit.to_static(Net())
        self.root_dir = os.path.join(get_saving_dir(), "wrapper")
        self.clean()

    def clean(self):
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)
        os.mkdir(self.root_dir)

    def test_export(self):
        x = paddle.randn([4, 4])
        x.stop_gradient = False
        out = self.net(x)
        self.check_export()

    def check_export(self):
        for prog_file in os.listdir(self.root_dir):
            if "forward" in prog_file:
                self.check_fwd(prog_file)
                return
            elif "backward" in prog_file:
                self.check_bwd(prog_file)
            else:
                raise RuntimeError("Not Support.")

    def check_fwd(self, prog_file):
        prog_info = [
            "pt_input_0",
            "pt_output_0",
            "pt_output_1",
            "pt_intermediate_0",
            "pt_intermediate_1",
            "pt_intermediate_2",
        ]
        path = os.path.join(self.root_dir, prog_file)
        with open(path, 'r') as f:
            content = f.readlines()
        index = 0
        for op_str in content:
            if "pd_op.data" in op_str or "pd_op.fetch" in op_str:
                self.assertIn(prog_info[index], op_str)
                index += 1

    def check_bwd(self, prog_file):
        prog_info = [
            "pt_input_6",
            "pt_input_5",
            "pt_input_4",
            "pt_input_3",
            "pt_input_2",
            "pt_input_1",
            "pt_input_0",
        ]
        path = os.path.join(self.root_dir, prog_file)
        with open(path, 'r') as f:
            content = f.readlines()
        index = 0
        for op_str in content:
            if "pd_op.data" in op_str or "pd_op.fetch" in op_str:
                self.assertIn(prog_info[index], op_str)
                index += 1


if __name__ == "__main__":
    unittest.main()
