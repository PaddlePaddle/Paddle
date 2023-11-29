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

import numpy as np

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

    def run_program(self, program, feed, fetch_list):
        paddle.enable_static()
        exe = paddle.static.Executor(paddle.CPUPlace())
        outs = exe._run_pir_impl(
            program,
            feed=feed,
            fetch_list=fetch_list,
            feed_var_name="feed",
            fetch_var_name='fetch',
            scope=None,
            return_numpy=True,
        )
        paddle.disable_static()
        return outs

    def check_export(self):
        for prog_file in os.listdir(self.root_dir):
            if "forward" in prog_file:
                self.check_fwd(prog_file)
            elif "backward" in prog_file:
                self.check_bwd(prog_file)
            else:
                raise RuntimeError("Not Support.")

    def check_fwd(self, prog_file):
        path = os.path.join(self.root_dir, prog_file)
        with open(path, 'r') as f:
            content = f.read()
        program = paddle.pir.parse_program(content)

        pt_input_0 = np.random.random([4, 4]).astype(np.float32)
        feed = {"pt_input_0": pt_input_0}
        fetch_list = [
            'pt_output_0',
            'pt_output_1',
            'pt_intermediate_0',
            'pt_intermediate_1',
            'pt_intermediate_2',
        ]
        outs = self.run_program(program, feed, fetch_list)

        self.assertEqual(len(outs), 5)
        out_shapes = [[], [4, 4], [4, 4], [4, 4], [4, 4]]
        for i, out in enumerate(outs):
            self.assertListEqual(list(out.shape), out_shapes[i])

    def check_bwd(self, prog_file):
        path = os.path.join(self.root_dir, prog_file)
        with open(path, 'r') as f:
            content = f.read()

        program = paddle.pir.parse_program(content)
        print(program)
        data = np.random.random([4, 4]).astype(np.float32)
        feed = {
            "pt_input_6": data,
            "pt_input_5": data,
            "pt_input_4": np.array(0.1).astype(np.float32),
            "pt_input_3": data,
            "pt_input_2": data,
            "pt_input_1": data,
            "pt_input_0": data,
        }
        fetch_list = []
        outs = self.run_program(program, feed, fetch_list)

        self.assertEqual(len(outs), 0)


# class TestSaveInferProg(TestSaveFwdBwdProg):

#     def test_export(self):
#         x = paddle.randn([4, 4])
#         self.net.eval()
#         out = self.net(x)
#         self.check_export()

#     def check_export(self):
#         for prog_file in os.listdir(self.root_dir):
#             breakpoint()
#             if "infer" in prog_file:
#                 self.check_infer(prog_file)
#             else:
#                 raise RuntimeError("Not Support.")

#     def check_infer(self, prog_file):
#         path = os.path.join(self.root_dir, prog_file)
#         with open(path, 'r') as f:
#             content = f.read()
#         program = paddle.pir.parse_program(content)

#         pt_input_0 = np.random.random([4,4]).astype(np.float32)
#         feed = {"pt_input_0": pt_input_0}
#         fetch_list = ['pt_output_0', 'pt_output_1']
#         outs = self.run_program(program, feed, fetch_list)

#         self.assertEqual(len(outs), 2)
#         out_shapes = [[], [4,4]]
#         for i, out in enumerate(outs):
#             self.assertListEqual(list(out.shape), out_shapes[i])

if __name__ == "__main__":
    unittest.main()
