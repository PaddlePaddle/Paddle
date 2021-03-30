#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import numpy as np
import unittest


class TestOpNameConflict(unittest.TestCase):
    def test_conflict(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                x = fluid.data(name="x", shape=[1], dtype='float32')
                y = fluid.data(name="y", shape=[1], dtype='float32')
                z = fluid.data(name="z", shape=[1], dtype='float32')

                m = fluid.layers.elementwise_add(x, y, name="add")
                n = fluid.layers.elementwise_add(y, z, name="add")
                p = m + n

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                m_v, n_v, p_v = exe.run(feed={
                    "x": np.ones((1), "float32") * 2,
                    "y": np.ones((1), "float32") * 3,
                    "z": np.ones((1), "float32") * 5
                },
                                        fetch_list=[m, n, p])

                self.assertEqual(m_v[0], 5.0)
                self.assertEqual(n_v[0], 8.0)
                self.assertEqual(p_v[0], 13.0)

    def test_layers(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
                ) else fluid.CPUPlace()
                exe = fluid.Executor(place)

                data = fluid.data(
                    name='data', shape=[None, 1, 2, 2], dtype='float32')
                tensor = fluid.data(
                    name='tensor', shape=[None, 32, 64], dtype='float32')
                x = fluid.data(
                    name='x', shape=[None, 1], dtype='float32', lod_level=1)

                input_scale = fluid.layers.create_parameter(
                    shape=[1],
                    dtype="float32",
                    default_initializer=fluid.initializer.Constant(2.0))
                input_bias = fluid.layers.create_parameter(
                    shape=[1],
                    dtype="float32",
                    default_initializer=fluid.initializer.Constant(0.5))
                out_affine = fluid.layers.affine_channel(
                    data, scale=input_scale, bias=input_bias)
                out_similarity = fluid.layers.similarity_focus(
                    input=data, axis=1, indexes=[0])
                position_tensor = fluid.layers.add_position_encoding(
                    input=tensor, alpha=1.0, beta=1.0)
                x_reversed = fluid.layers.sequence_reverse(x)

                exe.run(fluid.default_startup_program())
                test_program = fluid.default_main_program().clone(for_test=True)

                x_d = fluid.create_lod_tensor(
                    np.array([[1.1], [2.2], [3.3], [4.4]]).astype('float32'),
                    [[1, 3]], place)
                outs = exe.run(
                    test_program,
                    fetch_list=[
                        out_affine, out_similarity, position_tensor, x_reversed
                    ],
                    feed={
                        data.name: np.ones([1, 1, 2, 2]).astype('float32'),
                        tensor.name: np.ones([1, 32, 64]).astype('float32'),
                        x.name: x_d
                    },
                    return_numpy=False)


if __name__ == '__main__':
    unittest.main()
