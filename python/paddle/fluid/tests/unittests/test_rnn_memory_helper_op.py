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

import unittest

from paddle.fluid.framework import Program
from paddle.fluid.executor import Executor
import numpy as np
import paddle.fluid.core as core


class RNNMemoryHelperOpTest(unittest.TestCase):

    def setUp(self):
        self.program = Program()
        self.place = core.CPUPlace()

<<<<<<< HEAD
        self.X = self.program.global_block().create_var(name='X',
                                                        shape=[2, 3],
                                                        dtype='float32')
        self.Out = self.program.global_block().create_var(name='Out',
                                                          shape=[2, 3],
                                                          dtype='float32')
        self.program.global_block().append_op(type='rnn_memory_helper',
                                              inputs={"X": self.X},
                                              outputs={"Out": self.Out},
                                              attrs={})
=======
        self.X = self.program.global_block().create_var(
            name='X', shape=[2, 3], dtype='float32'
        )
        self.Out = self.program.global_block().create_var(
            name='Out', shape=[2, 3], dtype='float32'
        )
        self.program.global_block().append_op(
            type='rnn_memory_helper',
            inputs={"X": self.X},
            outputs={"Out": self.Out},
            attrs={},
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    def test_forward(self):
        x_np = np.random.normal(size=(2, 3)).astype("float32")
        self.feed_map = {'X': x_np}
        self.fetch_list = [self.Out]
        exe = Executor(self.place)
        out = exe.run(
            self.program, feed=self.feed_map, fetch_list=self.fetch_list
        )
        np.testing.assert_allclose(out[0], x_np, rtol=1e-05)


class RNNMemoryHelperGradOpTest(unittest.TestCase):

    def setUp(self):
        self.program = Program()
        self.place = core.CPUPlace()

        self.input_names = ['X', 'Out', 'Out@GRAD']
        self.input_vars = {
<<<<<<< HEAD
            name: self.program.global_block().create_var(name=name,
                                                         shape=[2, 3],
                                                         dtype='float32')
=======
            name: self.program.global_block().create_var(
                name=name, shape=[2, 3], dtype='float32'
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            for name in self.input_names
        }

        self.output_names = ['X@GRAD']
        self.output_vars = {
<<<<<<< HEAD
            name: self.program.global_block().create_var(name=name,
                                                         shape=[2, 3],
                                                         dtype='float32')
            for name in self.output_names
        }

        self.program.global_block().append_op(type='rnn_memory_helper_grad',
                                              inputs=self.input_vars,
                                              outputs=self.output_vars,
                                              attrs={})
=======
            name: self.program.global_block().create_var(
                name=name, shape=[2, 3], dtype='float32'
            )
            for name in self.output_names
        }

        self.program.global_block().append_op(
            type='rnn_memory_helper_grad',
            inputs=self.input_vars,
            outputs=self.output_vars,
            attrs={},
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    def test_backward(self):
        self.feed_map = {
            name: np.random.normal(size=(2, 3)).astype("float32")
            for name in self.input_names
        }
        self.fetch_list = [self.output_vars['X@GRAD']]

        exe = Executor(self.place)
        out = exe.run(
            self.program, feed=self.feed_map, fetch_list=self.fetch_list
        )
        np.isclose(out[0], self.feed_map['Out@GRAD'], rtol=1e-5)


class RNNMemoryHelperGradOpWithoutInputTest(unittest.TestCase):

    def setUp(self):
        self.program = Program()
        self.fake_program = Program()
        self.place = core.CPUPlace()

        self.input_names = ['X', 'Out']
        self.input_vars = {
<<<<<<< HEAD
            name: self.program.global_block().create_var(name=name,
                                                         shape=[2, 3],
                                                         dtype='float32')
=======
            name: self.program.global_block().create_var(
                name=name, shape=[2, 3], dtype='float32'
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            for name in self.input_names
        }
        self.input_vars[
            "Out@GRAD"
        ] = self.fake_program.global_block().create_var(
            name="Out@GRAD", shape=[2, 3], dtype='float32'
        )

        self.output_names = ['X@GRAD']
        self.output_vars = {
<<<<<<< HEAD
            name: self.program.global_block().create_var(name=name,
                                                         shape=[2, 3],
                                                         dtype='float32')
            for name in self.output_names
        }

        self.program.global_block().append_op(type='rnn_memory_helper_grad',
                                              inputs=self.input_vars,
                                              outputs=self.output_vars,
                                              attrs={})
=======
            name: self.program.global_block().create_var(
                name=name, shape=[2, 3], dtype='float32'
            )
            for name in self.output_names
        }

        self.program.global_block().append_op(
            type='rnn_memory_helper_grad',
            inputs=self.input_vars,
            outputs=self.output_vars,
            attrs={},
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    def test_backward(self):
        self.feed_map = {
            name: np.random.normal(size=(2, 3)).astype("float32")
            for name in ['X', 'Out']
        }
        self.fetch_list = [self.output_vars['X@GRAD']]

        exe = Executor(self.place)
<<<<<<< HEAD
        out = exe.run(self.program,
                      feed=self.feed_map,
                      fetch_list=self.fetch_list)
        self.assertTrue(
            np.allclose(out[0],
                        np.zeros(shape=(2, 3)).astype("float32"),
                        rtol=1e-5))
=======
        out = exe.run(
            self.program, feed=self.feed_map, fetch_list=self.fetch_list
        )
        np.testing.assert_allclose(
            out[0], np.zeros(shape=(2, 3)).astype('float32'), rtol=1e-05
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f


if __name__ == '__main__':
    unittest.main()
