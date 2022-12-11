#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import math
import sys

sys.path.append('..')
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.framework import Program, program_guard

paddle.enable_static()


class TestOneHotOp(OpTest):
    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.op_type = 'one_hot_v2'
        depth = 10
        depth_np = np.array(10).astype('int32')
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0])])

        out = np.zeros(shape=(np.product(x.shape), depth)).astype('float32')

        for i in range(np.product(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {'X': (x, x_lod), 'depth_tensor': depth_np}
        self.attrs = {'dtype': int(core.VarDesc.VarType.FP32)}
        self.outputs = {'Out': (out, x_lod)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestOneHotOp_attr(OpTest):
    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.op_type = 'one_hot_v2'
        depth = 10
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0]), 1])

        out = np.zeros(shape=(np.product(x.shape[:-1]), 1, depth)).astype(
            'float32'
        )

        for i in range(np.product(x.shape)):
            out[i, 0, x[i]] = 1.0

        self.inputs = {'X': (x, x_lod)}
        self.attrs = {'dtype': int(core.VarDesc.VarType.FP32), 'depth': depth}
        self.outputs = {'Out': (out, x_lod)}
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestOneHotOp_default_dtype(OpTest):
    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.op_type = 'one_hot_v2'
        depth = 10
        depth_np = np.array(10).astype('int32')
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0])])

        out = np.zeros(shape=(np.product(x.shape), depth)).astype('float32')

        for i in range(np.product(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {'X': (x, x_lod), 'depth_tensor': depth_np}
        self.attrs = {}
        self.outputs = {'Out': (out, x_lod)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestOneHotOp_default_dtype_attr(OpTest):
    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.op_type = 'one_hot_v2'
        depth = 10
        dimension = 12
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0]), 1])

        out = np.zeros(shape=(np.product(x.shape[:-1]), 1, depth)).astype(
            'float32'
        )

        for i in range(np.product(x.shape)):
            out[i, 0, x[i]] = 1.0

        self.inputs = {'X': (x, x_lod)}
        self.attrs = {'depth': depth}
        self.outputs = {'Out': (out, x_lod)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestOneHotOp_exception(unittest.TestCase):
    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.op_type = 'one_hot_v2'
        self.depth = 10
        self.place = core.CPUPlace()
        self.dimension = 12
        self.x = core.LoDTensor()
        x_lod = [[4, 1, 3, 3]]
        data = [np.random.randint(11, 20) for i in range(sum(x_lod[0]))]
        data = np.array(data).astype('int').reshape([sum(x_lod[0]), 1])
        self.x.set(data, self.place)
        self.x.set_recursive_sequence_lengths(x_lod)

    def test_check_output(self):
        program = Program()
        with program_guard(program):
            x = fluid.layers.data(
                name='x', shape=[self.dimension], dtype='float32', lod_level=1
            )
            block = program.current_block()
            one_hot_out = block.create_var(
                name="one_hot_out",
                type=core.VarDesc.VarType.LOD_TENSOR,
                dtype='float32',
            )
            block.append_op(
                type='one_hot',
                inputs={'X': x},
                attrs={'depth': self.depth},
                outputs={'Out': one_hot_out},
            )
            exe = fluid.Executor(self.place)

            def run():
                exe.run(
                    feed={'x': self.x},
                    fetch_list=[one_hot_out],
                    return_numpy=False,
                )

            self.assertRaises(ValueError, run)


class TestOneHotOpApi(unittest.TestCase):
    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True

    def test_api(self):
        depth = 10
        self._run(depth)

    def test_api_with_depthTensor(self):
        depth = fluid.layers.assign(input=np.array([10], dtype=np.int32))
        self._run(depth)

    def test_api_with_dygraph(self):
        depth = 10
        label = np.array(
            [np.random.randint(0, depth - 1) for i in range(6)]
        ).reshape([6, 1])
        with fluid.dygraph.guard():
            one_hot_label = fluid.one_hot(
                input=fluid.dygraph.to_variable(label), depth=depth
            )

            one_hot_label = paddle.nn.functional.one_hot(
                fluid.dygraph.to_variable(label), depth
            )
            # one_hot_label = paddle.nn.functional.one_hot(
            #     paddle.to_tensor(label), depth)

    def _run(self, depth):
        label = fluid.layers.data(name="label", shape=[1], dtype="int64")
        one_hot_label = fluid.one_hot(input=label, depth=depth)

        label_data = np.array(
            [np.random.randint(0, 10 - 1) for i in range(6)]
        ).reshape([6, 1])

        exe = fluid.Executor(self.place)
        exe.run(fluid.default_startup_program())
        ret = exe.run(
            feed={
                'label': label_data,
            },
            fetch_list=[one_hot_label],
            return_numpy=False,
        )


class BadInputTestOnehotV2(unittest.TestCase):
    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True

    def test_error(self):
        with fluid.program_guard(fluid.Program()):

            def test_bad_x():
                label = fluid.layers.data(
                    name="label",
                    shape=[4],
                    append_batch_size=False,
                    dtype="float32",
                )
                one_hot_label = fluid.one_hot(input=label, depth=4)

            self.assertRaises(TypeError, test_bad_x)


if __name__ == '__main__':
    unittest.main()
