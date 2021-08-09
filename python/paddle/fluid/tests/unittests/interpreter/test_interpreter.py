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

import unittest
import paddle
from paddle.fluid import core
from paddle.fluid.core import InterpreterCore

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
        inter_core = InterpreterCore(p, main_program.desc, startup_program.desc,
                                     core.Scope())

        out = inter_core.run({
            "a": np.ones(
                [2, 2], dtype="float32") * 2
        }, [c.name])
        for i in range(10):
            out = inter_core.run({
                "a": np.ones(
                    [2, 2], dtype="float32") * i
            }, [c.name])


class MultiStreamModelTestCase(unittest.TestCase):
    def setUp(self):
        assert core.is_compiled_with_cuda()
        self.place = paddle.CUDAPlace(0)
        paddle.seed(2020)

    def test_multi_stream(self):
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

        main_program = paddle.fluid.default_main_program()
        startup_program = paddle.fluid.default_startup_program()
        p = core.Place()
        p.set_place(self.place)

        # run with default Executor
        # exe = paddle.static.Executor(self.place)
        # exe.run(startup_program)
        # raw_out = exe.run(main_program, fetch_list=[mean])
        # print(raw_out)  # -0.9900857

        # raw_out = exe.run(main_program, fetch_list=[mean])
        # print(raw_out)  # -0.66547644

        inter_core = InterpreterCore(p, main_program.desc, startup_program.desc,
                                     core.Scope())
        out = inter_core.run({}, [mean.name])
        print(out)
        out = inter_core.run({}, [mean.name])
        print(out)
        # for i in range(10):
        #     if i == 4:
        #         core.nvprof_start()
        #         core.nvprof_enable_record_event()
        #     elif i==9:
        #         core.nvprof_stop()
        #     out = inter_core.run({}, [mean.name])
        #     print(out)


if __name__ == "__main__":
    unittest.main()
