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
from paddle.base.data_feeder import convert_dtype
from paddle.jit.dy2static.export_subgraph import get_saving_dir


class ProgramInfo:
    def __init__(self, program, feeds, fetchs):
        self.program = program
        # {name: [shape, dtype]}
        self.feeds = feeds
        # {name: shape}
        self.fetchs = fetchs

    def random_feeds(self):
        feed_dict = {}
        for name, info in self.feeds.items():
            data = np.random.random(info[0]).astype(convert_dtype(info[1]))
            feed_dict[name] = data

        return feed_dict

    def fetch_list(self):
        return list(self.fetchs.keys())


class Parser:
    def __init__(self):
        self.feed_op_name = 'pd_op.data'
        self.fetch_op_name = 'pd_op.fetch'

    def run(self, file):
        program = self.load_from(file)
        feeds = self.parse_feeds(program)
        fetchs = self.parse_fetchs(program)

        return ProgramInfo(program, feeds, fetchs)

    def load_from(self, file):
        with open(file, 'r') as f:
            content = f.read()

        return paddle.pir.parse_program(content)

    def parse_feeds(self, program):
        feeds = {}
        for op in program.global_block().ops:
            if op.name() == self.feed_op_name:
                in_val = op.result(0)
                # shape, dtype
                info = [in_val.shape, in_val.dtype]
                feeds[op.attrs()['name']] = info

        return feeds

    def parse_fetchs(self, program):
        fetchs = {}
        for op in program.global_block().ops:
            if op.name() == self.fetch_op_name:
                in_val = op.operand_source(0)
                fetchs[op.attrs()['name']] = in_val.shape

        return fetchs


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
        program_info = Parser().run(path)
        feed = program_info.random_feeds()
        fetch_list = program_info.fetch_list()
        outs = self.run_program(program_info.program, feed, fetch_list)

        self.assertEqual(len(outs), len(fetch_list))
        fetchs = program_info.fetchs
        for name, out in zip(fetch_list, outs):
            self.assertListEqual(list(out.shape), fetchs[name])

    def check_bwd(self, prog_file):
        path = os.path.join(self.root_dir, prog_file)
        program_info = Parser().run(path)
        feed = program_info.random_feeds()
        fetch_list = program_info.fetch_list()
        outs = self.run_program(program_info.program, feed, fetch_list)

        self.assertEqual(len(outs), len(fetch_list))


class TestSaveInferProg(TestSaveFwdBwdProg):
    def test_export(self):
        x = paddle.randn([4, 4])
        self.net.eval()
        out = self.net(x)
        self.check_export()

    def check_export(self):
        for prog_file in os.listdir(self.root_dir):
            if "infer" in prog_file:
                self.check_infer(prog_file)
            else:
                raise RuntimeError("Not Support.")

    def check_infer(self, prog_file):
        path = os.path.join(self.root_dir, prog_file)
        program_info = Parser().run(path)
        feed = program_info.random_feeds()
        fetch_list = program_info.fetch_list()
        outs = self.run_program(program_info.program, feed, fetch_list)

        self.assertEqual(len(outs), len(fetch_list))
        fetchs = program_info.fetchs
        for name, out in zip(fetch_list, outs):
            self.assertListEqual(list(out.shape), fetchs[name])


if __name__ == "__main__":
    unittest.main()
