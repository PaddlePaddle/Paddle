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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.layers.control_flow import lod_rank_table
from paddle.fluid import Program, program_guard
import numpy as np
import functools


def convert_to_offset(lod):
    offset = [[0] for i in lod]
    for i, level in enumerate(lod):
        for seq_len in level:
            offset[i].append(offset[i][-1] + seq_len)
    return offset


class TestReorderLoDTensor(unittest.TestCase):
    num_seq = 5
    # [name, shape, lod_level] pair indicating data info of source and target
    data_desc = (['input', [9], 0], ['ref', [5], 1])

    @classmethod
    def setUpClass(cls):
        cls.set_program()

    @classmethod
    def set_program(cls):
        dat = fluid.layers.data(
            name=cls.data_desc[0][0], shape=cls.data_desc[0][1]
        )
        dat.stop_gradient = False
        rank_dat = fluid.layers.data(
            name=cls.data_desc[1][0], shape=cls.data_desc[1][1]
        )
        table = lod_rank_table(rank_dat)
        new_dat = fluid.layers.reorder_lod_tensor_by_rank(
            x=dat, rank_table=table
        )
        loss = fluid.layers.reduce_sum(new_dat)
        fluid.backward.append_backward(loss=loss)
        cls.fetch_list = [new_dat, cls.data_desc[0][0] + '@GRAD']

    def run_program(self):
        outputs = []
        input_grads = []
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.set_inputs(place)
            exe = fluid.Executor(place)
            output, input_grad = exe.run(
                fluid.default_main_program(),
                feed=self.inputs,
                fetch_list=self.fetch_list,
                return_numpy=False,
            )
            outputs.append(output)
            input_grads.append(input_grad)
        self.actual_outputs = outputs
        self.actual_grads = input_grads

    def set_data(self):
        self.data = {}
        for desc in self.data_desc:
            data_name = desc[0]
            data_shape = desc[1]
            data_lod_level = desc[2]
            data_lod = []
            for i in range(data_lod_level):
                lod_level_i = np.random.randint(
                    low=1,
                    high=5,
                    size=self.num_seq
                    if i == 0
                    else sum(lod_level_i),  # noqa: F821
                ).tolist()
                data_lod.append(lod_level_i)
            data_value = np.random.random(
                size=[sum(data_lod[-1]) if data_lod else self.num_seq]
                + data_shape
            ).astype('float32')
            self.data[data_name] = (data_value, data_lod)

    def set_inputs(self, place):
        self.inputs = {}
        for desc in self.data_desc:
            tensor = fluid.Tensor()
            tensor.set(self.data[desc[0]][0], place)
            if self.data[desc[0]][1]:
                tensor.set_recursive_sequence_lengths(self.data[desc[0]][1])
            self.inputs[desc[0]] = tensor

    def reorder(self):
        level = 0
        # compute the rank_table according to ref_lod
        ref_lod = self.data[self.data_desc[1][0]][1][level]
        rank_table = []  # list of (index, length)
        for i in range(len(ref_lod)):
            rank_table.append((i, ref_lod[i]))
        rank_table = sorted(
            rank_table, key=functools.cmp_to_key(lambda x, y: y[1] - x[1])
        )

        # compute the input sequence info according to input_lod
        input_value, input_lod = self.data[self.data_desc[0][0]]
        offset_lod = convert_to_offset(input_lod)

        input_table = []  # list of (offset, length, sub_lod)
        if offset_lod:
            for i in range(len(offset_lod[level]) - 1):
                start_idx = i
                end_idx = i + 1
                sub_lod = []
                for lod_level_i in offset_lod[level:]:
                    sub_lod_i = []
                    for idx in range(start_idx, end_idx):
                        sub_lod_i.append(
                            lod_level_i[idx + 1] - lod_level_i[idx]
                        )
                    sub_lod.append(sub_lod_i)
                    start_idx = lod_level_i[start_idx]
                    end_idx = lod_level_i[end_idx]
                input_table.append((start_idx, end_idx - start_idx, sub_lod))
        else:
            input_table = [(i, 1, []) for i in range(len(rank_table))]

        # reorder by rank_table
        output_value = np.zeros_like(input_value)
        output_lod = []
        offset = 0
        for index, length in rank_table:
            input_seq_start = input_table[index][0]
            input_seq_len = input_table[index][1]
            input_seq_end = input_seq_start + input_seq_len
            output_value[offset : offset + input_seq_len] = input_value[
                input_seq_start:input_seq_end
            ]
            offset += input_seq_len

            input_seq_sub_lod = input_table[index][2]
            if len(output_lod) == 0:
                output_lod = [[] for i in input_seq_sub_lod]
            for i, level in enumerate(input_seq_sub_lod):
                output_lod[i].extend(level)
        return output_value, output_lod

    def test_reorder_lod_tensor(self):
        self.data_desc[0][-1] = 2  # input is lod_tensor
        self.set_data()
        self.run_program()
        # check output
        expect_output, expect_output_lod = self.reorder()
        for actual_output in self.actual_outputs:
            np.testing.assert_allclose(
                np.array(actual_output), expect_output, rtol=1e-05, atol=0.001
            )
            self.assertEqual(
                expect_output_lod, actual_output.recursive_sequence_lengths()
            )
        # check gradient
        expect_grad = np.ones_like(self.data[self.data_desc[0][0]][0])
        expect_grad_lod = self.data[self.data_desc[0][0]][1]
        for actual_grad in self.actual_grads:
            np.testing.assert_allclose(
                np.array(actual_grad), expect_grad, rtol=1e-05, atol=0.001
            )
            self.assertEqual(
                expect_grad_lod, actual_grad.recursive_sequence_lengths()
            )

    def test_reorder_tensor(self):
        self.data_desc[0][-1] = 0  # input is tensor
        self.set_data()
        self.run_program()
        # check output
        expect_output, expect_output_lod = self.reorder()
        for actual_output in self.actual_outputs:
            np.testing.assert_allclose(
                np.array(actual_output), expect_output, rtol=1e-05, atol=0.001
            )
            self.assertEqual(
                expect_output_lod, actual_output.recursive_sequence_lengths()
            )
        # check gradient
        expect_grad = np.ones_like(self.data[self.data_desc[0][0]][0])
        expect_grad_lod = self.data[self.data_desc[0][0]][1]
        for actual_grad in self.actual_grads:
            np.testing.assert_allclose(
                np.array(actual_grad), expect_grad, rtol=1e-05, atol=0.001
            )
            self.assertEqual(
                expect_grad_lod, actual_grad.recursive_sequence_lengths()
            )

        # compare outputs between LodTensors with explicit and implicit lod
        # use the same data but set the input lod explicitly
        input_lod = [[1] * len(self.data[self.data_desc[0][0]][0])]
        self.inputs[self.data_desc[0][0]].set_recursive_sequence_lengths(
            input_lod
        )
        # preserve the output of LodTensor with implicit lod to compare
        expect_outputs = [
            np.array(actual_output) for actual_output in self.actual_outputs
        ]
        self.run_program()
        for actual_output, expect_output in zip(
            self.actual_outputs, expect_outputs
        ):
            np.testing.assert_allclose(
                np.array(actual_output), expect_output, rtol=1e-05, atol=0.001
            )


class TestReorderLoDTensorError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):

            def test_Variable():
                # The input must be Variable.
                x1 = np.array([0.9383, 0.1983, 3.2, 1.2]).astype("float64")
                table1 = np.array([0.9383, 0.1983, 3.2, 1.2]).astype("float64")
                new_dat = fluid.layers.reorder_lod_tensor_by_rank(
                    x=x1, rank_table=table1
                )

            self.assertRaises(TypeError, test_Variable)

            def test_type():
                x2 = fluid.layers.data(name='x1', shape=[4], dtype='float32')
                table2 = fluid.layers.data(
                    name='table2', shape=[4], dtype='int32'
                )
                new_dat2 = fluid.layers.reorder_lod_tensor_by_rank(
                    x=x2, rank_table=table2
                )

            self.assertRaises(TypeError, test_type)


if __name__ == '__main__':
    unittest.main()
