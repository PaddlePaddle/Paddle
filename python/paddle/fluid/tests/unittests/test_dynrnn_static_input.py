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
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.backward import append_backward
import paddle.fluid.framework as framework
from paddle.fluid.framework import Program, switch_main_program
import bisect
import numpy as np

fluid.default_startup_program().random_seed = 1
np.random.seed(1)


class TestDyRnnStaticInput(unittest.TestCase):

    def setUp(self):
        self._delta = 0.005
        self._max_sequence_len = 3
        self._program = Program()
        switch_main_program(self._program)
        self.output_dim = 10
        self.place = core.CPUPlace()
        self.prepare_x_tensor()
        self.prepare_static_input_tensor()
        self.exe = fluid.Executor(self.place)

    def prepare_x_tensor(self):
        self.x_tensor_dim = 10
        lod = [[2, 1, 3]]
        shape = [sum(lod[0]), self.x_tensor_dim]
        self.x_tensor_data = np.random.random(shape).astype('float32')
        self.x_tensor = core.LoDTensor()
        self.x_tensor.set_recursive_sequence_lengths(lod)
        self.x_tensor.set(self.x_tensor_data, self.place)

    def prepare_static_input_tensor(self):
        self.static_input_tensor_dim = 4
        lod = [[1, 2, 3]]
        shape = [sum(lod[0]), self.static_input_tensor_dim]
        self.static_input_data = np.random.random(shape).astype('float32')
        self.static_input_tensor = core.LoDTensor()
        self.static_input_tensor.set_recursive_sequence_lengths(lod)
        self.static_input_tensor.set(self.static_input_data, self.place)

    def fetch_value(self, var):
        fetch_outs = self.exe.run(feed={
            'x_tensor':
            self.x_tensor,
            'static_input_tensor':
            self.static_input_tensor
        },
                                  fetch_list=[var],
                                  return_numpy=False)
        return self._lodtensor_to_ndarray(fetch_outs[0])

    def _lodtensor_to_ndarray(self, lod_tensor):
        dims = lod_tensor.shape()
        ndarray = np.zeros(shape=dims).astype('float32')
        for i in range(np.product(dims)):
            ndarray.ravel()[i] = lod_tensor._get_float_element(i)
        return ndarray, lod_tensor.recursive_sequence_lengths()

    def build_graph(self, only_forward=False):
        x_tensor = fluid.layers.data(name='x_tensor',
                                     shape=[self.x_tensor_dim],
                                     dtype='float32',
                                     lod_level=1)
        x_tensor.stop_gradient = False

        static_input_tensor = fluid.layers.data(
            name='static_input_tensor',
            shape=[self.static_input_tensor_dim],
            dtype='float32',
            lod_level=1)
        static_input_tensor.stop_gradient = False

        if only_forward:
            static_input_out_array = self._program.global_block().create_var(
                name='static_input_out_array',
                type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                dtype='float32')
            static_input_out_array.stop_gradient = True

        rnn = fluid.layers.DynamicRNN()
        with rnn.block():
            step_x = rnn.step_input(x_tensor)
            step_static_input = rnn.static_input(static_input_tensor)
            if only_forward:
                fluid.layers.array_write(x=step_static_input,
                                         i=rnn.step_idx,
                                         array=static_input_out_array)
            last = fluid.layers.sequence_pool(input=step_static_input,
                                              pool_type='last')
            projected = fluid.layers.fc(input=[step_x, last],
                                        size=self.output_dim)
            rnn.output(projected)

        if only_forward:
            static_input_step_outs = []
            step_idx = fluid.layers.fill_constant(shape=[1],
                                                  dtype='int64',
                                                  value=0)
            step_idx.stop_gradient = True

            for i in range(self._max_sequence_len):
                step_out = fluid.layers.array_read(static_input_out_array,
                                                   step_idx)
                step_out.stop_gradient = True
                static_input_step_outs.append(step_out)
                fluid.layers.increment(x=step_idx, value=1.0, in_place=True)

        if only_forward:
            return static_input_step_outs

        last = fluid.layers.sequence_pool(input=rnn(), pool_type='last')
        loss = paddle.mean(last)
        append_backward(loss)
        static_input_grad = self._program.global_block().var(
            framework.grad_var_name('static_input_tensor'))
        return static_input_grad, loss

    def get_expected_static_step_outs(self):
        x_lod = self.x_tensor.recursive_sequence_lengths()
        x_seq_len = x_lod[0]
        x_seq_len_sorted = sorted(x_seq_len)
        x_sorted_indices = np.argsort(x_seq_len)[::-1]

        static_lod = self.static_input_tensor.recursive_sequence_lengths()
        static_sliced = []
        cur_offset = 0
        for i in range(len(static_lod[0])):
            static_sliced.append(
                self.static_input_data[cur_offset:(cur_offset +
                                                   static_lod[0][i])])
            cur_offset += static_lod[0][i]
        static_seq_len = static_lod[0]
        static_reordered = []
        for i in range(len(x_sorted_indices)):
            static_reordered.extend(static_sliced[x_sorted_indices[i]].tolist())
        static_seq_len_reordered = [
            static_seq_len[x_sorted_indices[i]]
            for i in range(len(x_sorted_indices))
        ]

        static_step_outs = []
        static_step_lods = []

        for i in range(self._max_sequence_len):
            end = len(x_seq_len) - bisect.bisect_left(x_seq_len_sorted, i + 1)
            lod = []
            total_len = 0
            for i in range(end):
                lod.append(static_seq_len_reordered[i])
                total_len += lod[-1]
            static_step_lods.append([lod])
            end = total_len
            static_step_outs.append(
                np.array(static_reordered[:end]).astype('float32'))

        return static_step_outs, static_step_lods

    def test_step_out(self):
        static_step_outs = self.build_graph(only_forward=True)
        self.exe.run(framework.default_startup_program())
        expected_outs, expected_lods = self.get_expected_static_step_outs()
        for i in range(self._max_sequence_len):
            step_out, lod = self.fetch_value(static_step_outs[i])
            np.testing.assert_allclose(step_out, expected_outs[i], rtol=1e-05)
            np.testing.assert_allclose(lod, expected_lods[i], rtol=1e-05)

    def test_network_gradient(self):
        static_input_grad, loss = self.build_graph()
        self.exe.run(framework.default_startup_program())

        actual_gradients, actual_lod = self.fetch_value(static_input_grad)

        static_input_shape = self.static_input_tensor.shape()
        numeric_gradients = np.zeros(shape=static_input_shape).astype('float32')
        # calculate numeric gradients
        tensor_size = np.product(static_input_shape)
        for i in range(tensor_size):
            origin = self.static_input_tensor._get_float_element(i)
            x_pos = origin + self._delta
            self.static_input_tensor._set_float_element(i, x_pos)
            y_pos = self.fetch_value(loss)[0][0]
            x_neg = origin - self._delta
            self.static_input_tensor._set_float_element(i, x_neg)
            y_neg = self.fetch_value(loss)[0][0]
            self.static_input_tensor._set_float_element(i, origin)
            numeric_gradients.ravel()[i] = (y_pos - y_neg) / self._delta / 2
        np.testing.assert_allclose(actual_gradients,
                                   numeric_gradients,
                                   rtol=0.001)
        np.testing.assert_allclose(
            actual_lod,
            self.static_input_tensor.recursive_sequence_lengths(),
            rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
