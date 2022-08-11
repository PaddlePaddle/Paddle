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

from __future__ import print_function

import unittest
import paddle
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestIndexAddOp(unittest.TestCase):

    def setUp(self):
        self.setType()
        self.setPlace()
        self.config()
        self.generate_input_data()

        self.index_shape = tuple([self.index_size])

        self.rtol = 1e-5
        self.atol = 1e-2
        if self.x_type is np.float16:
            self.atol = 1e-1

    def setType(self):
        self.x_type = np.float32
        self.index_type = np.int32

    def setPlace(self):
        self.place = "cpu"

    def config(self):
        self.axis = 0
        self.x_shape = (100, 5)
        self.index_size = 20
        self.add_value_shape = (20, 5)

    def generate_input_data(self):
        axis = self.axis
        if self.axis < 0:
            axis = self.axis + len(self.x_shape)

        self.x_np = np.random.random(self.x_shape).astype(self.x_type)
        self.add_value_np = np.random.random(self.add_value_shape).astype(
            self.x_type)
        self.index_np = np.random.randint(low=0,
                                          high=self.x_shape[axis],
                                          size=self.index_size).astype(
                                              self.index_type)

    def compute_index_add_ref(self):
        axis = self.axis
        if self.axis < 0:
            axis = self.axis + len(self.x_shape)
        if axis != 0:
            outer_loop = np.prod(self.x_shape[:axis]).astype(int)
            print("outer_loop: ", outer_loop)
            x_reshape = [outer_loop] + list(self.x_shape[axis:])
            print("x_reshape: ", x_reshape)
            x_np_reshape = np.reshape(self.x_np, tuple(x_reshape))

            add_value_reshape = [
                np.prod(self.add_value_shape[:axis]).astype(int)
            ] + list(self.add_value_shape[axis:])
            print("add_value_reshape: ", add_value_reshape)

            add_value_np_reshape = np.reshape(self.add_value_np,
                                              tuple(add_value_reshape))
        else:
            x_np_reshape = self.x_np
            add_value_np_reshape = self.add_value_np
        out_np = x_np_reshape

        if axis != 0:
            for i in range(outer_loop):
                for j in range(self.index_size):
                    out_np[i, self.index_np[j]] += add_value_np_reshape[i, j]
        else:
            for j in range(self.index_size):
                out_np[self.index_np[j]] += add_value_np_reshape[j]
        ref_out = np.reshape(out_np, self.x_shape)
        return ref_out

    def run_imperative(self):
        paddle.device.set_device(self.place)
        input_tensor = paddle.to_tensor(self.x_np)
        index = paddle.to_tensor(self.index_np)
        add_value = paddle.to_tensor(self.add_value_np)

        out = paddle.index_add(input_tensor, index, add_value, axis=self.axis)
        ref_out = self.compute_index_add_ref()

        np.testing.assert_allclose(ref_out,
                                   out.numpy(),
                                   rtol=self.rtol,
                                   atol=self.atol)

    def run_static(self):
        x = paddle.static.data(name='X', shape=self.x_shape, dtype=self.x_type)
        index = paddle.static.data(name='Index',
                                   shape=self.index_shape,
                                   dtype=self.index_type)
        add_value = paddle.static.data(name='AddValue',
                                       shape=self.add_value_shape,
                                       dtype=self.x_type)

        out = paddle.index_add(x, index, add_value, self.axis)

        if self.place == "cpu":
            place = paddle.CPUPlace()
        elif self.place == "gpu":
            place = paddle.CUDAPlace(0)
        else:
            raise TypeError(
                "paddle.index_add api only support cpu and gpu device now.")

        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        res = exe.run(paddle.static.default_main_program(),
                      feed={
                          "X": self.x_np,
                          "Index": self.index_np,
                          "AddValue": self.add_value_np,
                      },
                      fetch_list=[out.name],
                      return_numpy=False)
        return res

    def test_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(Program()):
            out = self.run_static()
        ref_out = self.compute_index_add_ref()
        np.testing.assert_allclose(ref_out,
                                   np.array(out[0]),
                                   rtol=self.rtol,
                                   atol=self.atol)

    def test_dynamic(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        self.run_imperative()


class TestIndexAddOpMoreType(TestIndexAddOp):

    def setType(self):
        self.x_type = np.float16
        self.index_type = np.int64


class TestIndexAdOpCase2(TestIndexAddOp):

    def config(self):
        self.axis = 1
        self.x_shape = (100, 100, 5)
        self.index_size = 20
        self.add_value_shape = (100, 20, 5)


class TestIndexAdOpCase3(TestIndexAddOp):

    def config(self):
        self.axis = 2
        self.x_shape = (100, 100, 25)
        self.index_size = 20
        self.add_value_shape = (100, 100, 20)


class TestIndexAdOpCase4(TestIndexAddOp):

    def config(self):
        self.axis = 0
        self.x_shape = (10, )
        self.index_size = 4
        self.add_value_shape = (4, )


class TestIndexAdOpCase5(TestIndexAddOp):

    def config(self):
        self.axis = -1
        self.x_shape = (10, 10)
        self.index_size = 4
        self.add_value_shape = (10, 4)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestIndexAdOpGPU(TestIndexAddOp):

    def setPlace(self):
        self.place = "gpu"


if __name__ == '__main__':
    unittest.main()
