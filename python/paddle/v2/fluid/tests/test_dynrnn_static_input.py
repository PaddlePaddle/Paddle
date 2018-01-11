import unittest
import paddle.v2 as paddle
import paddle.v2.fluid.core as core
import paddle.v2.fluid as fluid
from paddle.v2.fluid.backward import append_backward
import paddle.v2.fluid.framework as framework
from paddle.v2.fluid.framework import Program, switch_main_program
import bisect
import numpy as np

fluid.default_startup_program().random_seed = 0
np.random.seed(0)


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
        lod = [[0, 2, 3, 6]]
        shape = [lod[0][-1], self.x_tensor_dim]
        self.x_tensor_data = np.random.random(shape).astype('float32')
        self.x_tensor = core.LoDTensor()
        self.x_tensor.set_lod(lod)
        self.x_tensor.set(self.x_tensor_data, self.place)

    def prepare_static_input_tensor(self):
        self.static_input_tensor_dim = 4
        lod = [[0, 1, 3, 6]]
        shape = [lod[0][-1], self.static_input_tensor_dim]
        self.static_input_data = np.random.random(shape).astype('float32')
        self.static_input_tensor = core.LoDTensor()
        self.static_input_tensor.set_lod(lod)
        self.static_input_tensor.set(self.static_input_data, self.place)

    def fetch_value(self, var):
        fetch_outs = self.exe.run(feed={
            'x_tensor': self.x_tensor,
            'static_input_tensor': self.static_input_tensor
        },
                                  fetch_list=[var],
                                  return_numpy=False)
        return self._lodtensor_to_ndarray(fetch_outs[0])

    def _lodtensor_to_ndarray(self, lod_tensor):
        dims = lod_tensor.get_dims()
        ndarray = np.zeros(shape=dims).astype('float32')
        for i in xrange(np.product(dims)):
            ndarray.ravel()[i] = lod_tensor.get_float_element(i)
        return ndarray

    def build_graph(self, only_forward=False):
        x_tensor = fluid.layers.data(
            name='x_tensor',
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
                fluid.layers.array_write(
                    x=step_static_input,
                    i=rnn.step_idx,
                    array=static_input_out_array)
            last = fluid.layers.sequence_pool(
                input=step_static_input, pool_type='last')
            projected = fluid.layers.fc(input=[step_x, last],
                                        size=self.output_dim)
            rnn.output(projected)

        if only_forward:
            static_input_step_outs = []
            step_idx = fluid.layers.fill_constant(
                shape=[1], dtype='int64', value=0)
            step_idx.stop_gradient = True

            for i in xrange(self._max_sequence_len):
                step_out = fluid.layers.array_read(static_input_out_array,
                                                   step_idx)
                step_out.stop_gradient = True
                static_input_step_outs.append(step_out)
                fluid.layers.increment(x=step_idx, value=1.0, in_place=True)

        if only_forward:
            return static_input_step_outs

        last = fluid.layers.sequence_pool(input=rnn(), pool_type='last')
        loss = fluid.layers.mean(x=last)
        append_backward(loss)
        static_input_grad = self._program.global_block().var(
            framework.grad_var_name('static_input_tensor'))
        return static_input_grad, loss

    def get_seq_len_from_lod(self, lod):
        return [lod[0][i + 1] - lod[0][i] for i in xrange(len(lod[0]) - 1)]

    def get_expected_static_step_outs(self):
        x_lod = self.x_tensor.lod()
        x_seq_len = self.get_seq_len_from_lod(x_lod)
        x_seq_len_sorted = sorted(x_seq_len)
        x_sorted_indices = np.argsort(x_seq_len)[::-1]

        static_lod = self.static_input_tensor.lod()
        static_sliced = [
            self.static_input_data[static_lod[0][i]:static_lod[0][i + 1]]
            for i in xrange(len(static_lod[0]) - 1)
        ]
        static_seq_len = self.get_seq_len_from_lod(static_lod)
        static_reordered = []
        for i in xrange(len(x_sorted_indices)):
            static_reordered.extend(static_sliced[x_sorted_indices[i]].tolist())
        static_seq_len_reordered = [
            static_seq_len[x_sorted_indices[i]]
            for i in xrange(len(x_sorted_indices))
        ]

        static_step_outs = []

        for i in xrange(self._max_sequence_len):
            end = len(x_seq_len) - bisect.bisect_left(x_seq_len_sorted, i + 1)
            end = sum(static_seq_len_reordered[:end])
            static_step_outs.append(
                np.array(static_reordered[:end]).astype('float32'))

        return static_step_outs

    def test_step_out(self):
        static_step_outs = self.build_graph(only_forward=True)
        self.exe.run(framework.default_startup_program())
        expected_step_outs = self.get_expected_static_step_outs()
        for i in xrange(self._max_sequence_len):
            step_out = self.fetch_value(static_step_outs[i])
            self.assertTrue(np.allclose(step_out, expected_step_outs[i]))

    def test_network_gradient(self):
        pass  #still have bug (seed doesn't work)
        '''
        static_input_grad, loss = self.build_graph()
        self.exe.run(framework.default_startup_program())

        actual_gradients = self.fetch_value(static_input_grad)

        static_input_shape = self.static_input_tensor.get_dims()
        numeric_gradients = np.zeros(shape=static_input_shape).astype('float32')
        #print(actual_gradient)
        print(actual_gradients)
        # calculate numeric gradients
        tensor_size = np.product(static_input_shape)
        for i in xrange(tensor_size):
            origin = self.static_input_tensor.get_float_element(i)
            x_pos = origin + self._delta
            self.static_input_tensor.set_float_element(i, x_pos)
            y_pos = self.fetch_value(loss)[0]
            x_neg = origin - self._delta
            self.static_input_tensor.set_float_element(i, x_neg)
            y_neg = self.fetch_value(loss)[0]
            self.static_input_tensor.set_float_element(i, origin)
            numeric_gradients.ravel()[i] = (y_pos - y_neg) / self._delta / 2

        print(numeric_gradients)
        '''


if __name__ == '__main__':
    unittest.main()
