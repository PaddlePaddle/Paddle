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

import sys
import unittest
import paddle
from paddle.fluid import core
from paddle.fluid.core import StandaloneExecutor

import numpy as np

paddle.enable_static()


class LinearTestCase(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()

    def test_interp_base(self):
        a = paddle.static.data(name="a", shape=[2, 2], dtype='float32')
        b = paddle.ones([2, 2]) * 2
        t = paddle.static.nn.fc(a, 2)
        c = t + b

        main_program = paddle.fluid.default_main_program()
        startup_program = paddle.fluid.default_startup_program()
        p = core.Place()
        p.set_place(self.place)
        standaloneexecutor = StandaloneExecutor(p, startup_program.desc,
                                                main_program.desc, core.Scope())

        out = standaloneexecutor.run({
            "a": np.ones(
                [2, 2], dtype="float32") * 2
        }, [c.name])
        for i in range(10):
            out = standaloneexecutor.run({
                "a": np.ones(
                    [2, 2], dtype="float32") * i
            }, [c.name])

        for i in range(10):
            out = standaloneexecutor.run({
                "a": np.ones(
                    [2, 2], dtype="float32") * i
            }, [a.name, c.name])

        # test for cost_info
        cost_info = standaloneexecutor.dry_run({
            "a": np.ones(
                [2, 2], dtype="float32") * i
        })
        self.check_cost_info(cost_info)

    def check_cost_info(self, cost_info):
        if core.is_compiled_with_cuda():
            # self.assertEqual(cost_info.host_memory_bytes(), 16)
            self.assertGreater(cost_info.device_memory_bytes(), 0)
            self.assertGreaterEqual(cost_info.device_total_memory_bytes(),
                                    cost_info.device_memory_bytes())
        else:
            self.assertGreater(cost_info.host_memory_bytes(), 0)
            self.assertEqual(cost_info.device_memory_bytes(), 0)


class MultiStreamModelTestCase(unittest.TestCase):
    def setUp(self):
        self.iter_n = 2
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()

    def build_program(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        with paddle.static.program_guard(main_program, startup_program):
            with paddle.static.device_guard('cpu'):
                data = paddle.ones([4, 64], dtype='float32', name='data')

            # data -> [memcpy_h2d] -> data' -> [matmul] -> out ->[add] -> add_out
            with paddle.static.device_guard('gpu'):
                weight = paddle.randn([64, 64], name='weight')  # gpu
                matmul_out = paddle.matmul(
                    data, weight, name='matmul_out')  # gpus
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

    def test_multi_stream(self):
        ground_truths = self.run_raw_executor()
        res = self.run_new_executor()
        for gt, out in zip(ground_truths, res):
            self.assertEqual(gt[0], out[0])

    def run_raw_executor(self):
        paddle.seed(2020)
        main_program, startup_program, fetch_list = self.build_program()

        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)

        outs = []
        for i in range(self.iter_n):
            outs.append(exe.run(main_program, fetch_list=fetch_list))
        return outs

    def run_new_executor(self):
        paddle.seed(2020)
        main_program, startup_program, fetch_list = self.build_program()
        fetch_list = [x.name for x in fetch_list]

        p = core.Place()
        p.set_place(self.place)
        inter_core = StandaloneExecutor(p, startup_program.desc,
                                        main_program.desc, core.Scope())
        outs = []
        for i in range(self.iter_n):
            outs.append(
                np.array(inter_core.run({}, fetch_list)._move_to_list()[0]))
        return outs


if __name__ == "__main__":
    unittest.main()
