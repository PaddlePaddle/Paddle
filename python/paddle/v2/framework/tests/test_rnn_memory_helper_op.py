import unittest

from paddle.v2.framework.framework import Program
from paddle.v2.framework.executor import Executor
from paddle.v2.framework.backward import append_backward_ops
import numpy as np
import paddle.v2.framework.core as core


def create_tensor(np_data, place):
    tensor = core.LoDTensor()
    tensor.set(np_data, place)
    return tensor


class RNNMemoryHelperOpTest(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.place = core.CPUPlace()

        self.X = self.program.global_block().create_var(
            name='X', shape=[2, 3], dtype='float32')
        self.Out = self.program.global_block().create_var(
            name='Out', shape=[2, 3], dtype='float32')
        self.program.global_block().append_op(
            type='rnn_memory_helper',
            inputs={"X": self.X},
            outputs={"Out": self.Out},
            attrs={})

    def test_forward(self):
        x_np = np.random.normal(size=(2, 3)).astype("float32")
        self.feed_map = {'X': create_tensor(x_np, self.place)}
        self.fetch_list = [self.Out]
        exe = Executor(self.place)
        out = exe.run(self.program,
                      feed=self.feed_map,
                      fetch_list=self.fetch_list)
        np.isclose(np.array(out[0]), x_np, rtol=1e-5)


class RNNMemoryHelperGradOpTest(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.place = core.CPUPlace()

        self.input_names = ['X', 'Out', 'Out@GRAD']
        self.input_vars = {
            name: self.program.global_block().create_var(
                name=name, shape=[2, 3], dtype='float32')
            for name in self.input_names
        }

        self.output_names = ['X@GRAD']
        self.output_vars = {
            name: self.program.global_block().create_var(
                name=name, shape=[2, 3], dtype='float32')
            for name in self.output_names
        }

        self.program.global_block().append_op(
            type='rnn_memory_helper_grad',
            inputs=self.input_vars,
            outputs=self.output_vars,
            attrs={})

    def test_backward(self):
        self.feed_map = {
            name: create_tensor(
                np.random.normal(size=(2, 3)).astype("float32"), self.place)
            for name in self.input_names
        }
        self.fetch_list = [self.output_vars['X@GRAD']]

        exe = Executor(self.place)
        out = exe.run(self.program,
                      feed=self.feed_map,
                      fetch_list=self.fetch_list)
        np.isclose(np.array(out[0]), self.feed_map['Out@GRAD'], rtol=1e-5)


class RNNMemoryHelperGradOpWithoutInputTest(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.fake_program = Program()
        self.place = core.CPUPlace()

        self.input_names = ['X', 'Out']
        self.input_vars = {
            name: self.program.global_block().create_var(
                name=name, shape=[2, 3], dtype='float32')
            for name in self.input_names
        }
        self.input_vars["Out@GRAD"] = \
            self.fake_program.global_block().create_var(
                name="Out@GRAD", shape=[2, 3], dtype='float32')

        self.output_names = ['X@GRAD']
        self.output_vars = {
            name: self.program.global_block().create_var(
                name=name, shape=[2, 3], dtype='float32')
            for name in self.output_names
        }

        self.program.global_block().append_op(
            type='rnn_memory_helper_grad',
            inputs=self.input_vars,
            outputs=self.output_vars,
            attrs={})

    def test_backward(self):
        self.feed_map = {
            name: create_tensor(
                np.random.normal(size=(2, 3)).astype("float32"), self.place)
            for name in ['X', 'Out']
        }
        self.fetch_list = [self.output_vars['X@GRAD']]

        exe = Executor(self.place)
        out = exe.run(self.program,
                      feed=self.feed_map,
                      fetch_list=self.fetch_list)
        np.isclose(
            np.array(out[0]),
            np.zeros(shape=(2, 3)).astype("float32"),
            rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
