# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

os.environ['FLAGS_use_stream_safe_cuda_allocator'] = "true"
import unittest

import numpy as np
from utils import static_guard

import paddle
from paddle.base import core

paddle.enable_static()


def build_program():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        with paddle.static.device_guard('cpu'):
            data = paddle.ones([4, 64], dtype='float32', name='data')

        # data -> [memcpy_h2d] -> data' -> [matmul] -> out ->[add] -> add_out
        with paddle.static.device_guard('gpu'):
            weight = paddle.randn([64, 64], name='weight')  # gpu
            matmul_out = paddle.matmul(data, weight, name='matmul_out')  # gpus
            bias = paddle.ones([4, 64], dtype='float32', name='bias')
            add_out = paddle.add(matmul_out, bias, name='add_out')

        # add_out -> [memcpy_d2h] -> add_out' -> [sub] -> sub_out -> [tanh] -> tanh_out
        with paddle.static.device_guard('cpu'):
            sub_out = paddle.subtract(add_out, data, name='sub_out')
            tanh_out = paddle.tanh(sub_out, name='tanh_out')

        with paddle.static.device_guard('gpu'):
            bias_1 = paddle.add(bias, sub_out, name='bias_1')
            out_before = paddle.tanh(bias_1, name='out_before')
            out_last = paddle.subtract(tanh_out, data, name='out_last')

            out = paddle.add(out_before, out_last, name='out')
            mean = paddle.mean(out, name='mean_out')

    return main_program, startup_program, [mean]


class SwitchExecutorInterfaceWithFeed(unittest.TestCase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.iter_run = 2

    def build_program(self, is_double=False):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            a = paddle.static.data(name="a", shape=[2, 2], dtype='float32')
            b = paddle.ones([2, 2]) * 2
            t = paddle.static.nn.fc(a, 2)
            c = t + b
            if is_double:
                c = c + c

        return main_program, startup_program, [c]

    def _run(
        self,
        feed,
        use_str=False,
        is_double=False,
        add_wrong_fetch=False,
        use_compiled=False,
    ):
        paddle.seed(2020)

        main_program, startup_program, fetch_vars = self.build_program(
            is_double
        )

        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)

        if use_compiled:
            main_program = paddle.static.CompiledProgram(main_program)

        if use_str:  # test for fetch name
            fetch_vars = [x.name for x in fetch_vars]
        if add_wrong_fetch:  # test for wrong fetch type
            fetch_vars.append(1123)
        outs = []
        for i in range(self.iter_run):
            out = exe.run(main_program, feed=feed, fetch_list=fetch_vars)[0]

            outs.append(out)

        return outs

    def run_dygraph(self, feed):
        def run_once(is_double):
            paddle.seed(2020)
            a = feed['a']
            a = paddle.to_tensor(a, dtype='float32')
            b = paddle.ones([2, 2]) * 2
            t = paddle.nn.Linear(2, 2)(a)
            c = t + b
            if is_double:
                c = c + c
            return c.numpy()

        out1 = []
        for i in range(self.iter_run):
            out1.append(run_once(False))
        out2 = []
        for i in range(self.iter_run):
            out2.append(run_once(True))
        return [out1, out2]

    def run_new_executor(self, feed, use_compiled=False):
        # run construct program 1
        out1 = self._run(
            feed, use_str=False, is_double=False, use_compiled=use_compiled
        )
        # run construct program 2 with same executor
        out2 = self._run(
            feed, use_str=True, is_double=True, use_compiled=use_compiled
        )

        return [out1, out2]

    def test_with_feed(self):
        data = np.ones([2, 2], dtype="float32")
        feed = {"a": data, 'fake_input': data}

        with static_guard():
            res = self.run_new_executor(feed)
        with paddle.base.dygraph.guard():
            gt = self.run_dygraph(feed)
        for x, y in zip(gt, res):
            np.testing.assert_array_equal(x, y)

    def test_with_error(self):
        feed = [{'a': np.ones([2, 2], dtype="float32")}]

        with self.assertRaises(TypeError):
            self._run(feed[0], add_wrong_fetch=True)

    def test_empty_program(self):
        program = paddle.static.Program()
        exe = paddle.static.Executor(self.place)
        for i in range(10):
            out = exe.run()  # old executor

        for i in range(10):
            print(i, flush=1)
            out = exe.run(program, feed=None)


if __name__ == "__main__":
    unittest.main()
