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
import shutil
import argparse
import unittest
os.environ['FLAGS_enable_pir_api'] = "1"
os.environ['FLAGS_cinn_new_group_scheduler']="1" 
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
                shape = []
                for s in in_val.shape:
                    if s == -1:
                        s = 1
                    shape.append(s)
                info = [shape, in_val.dtype]
                feeds[op.attrs()['name']] = info

        return feeds

    def parse_fetchs(self, program):
        fetchs = {}
        for op in program.global_block().ops:
            if op.name() == self.fetch_op_name:
                in_val = op.operand_source(0)
                fetchs[op.attrs()['name']] = in_val.shape

        return fetchs



class TestTask(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.file_path = args.file_path
    
    def test_phi(self):
        self.check_infer(enable_cinn=False)

    def test_cinn(self):
        self.check_infer(enable_cinn=True)

    def check_infer(self, enable_cinn):
        program_info = Parser().run(self.file_path)
        feed = program_info.random_feeds()
        fetch_list = program_info.fetch_list()
        self.run_program(program_info.program, feed, fetch_list, enable_cinn)
    
    def run_program(self, program, feed, fetch_list, enable_cinn):
        if enable_cinn:
            paddle.base.libpaddle.pir.apply_pir_pass(program)
        exe = paddle.static.Executor(paddle.CUDAPlace(2))
        outs = exe._run_pir_impl(
            program,
            feed=feed,
            fetch_list=fetch_list,
            feed_var_name="feed",
            fetch_var_name='fetch',
            scope=None,
            return_numpy=True,
        )
        return outs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='input file', dest='file_path')
    parser.add_argument('unittest_args', nargs='*')
    args = parser.parse_args()
    sys.argv[1:] = args.unittest_args
    unittest.main()
