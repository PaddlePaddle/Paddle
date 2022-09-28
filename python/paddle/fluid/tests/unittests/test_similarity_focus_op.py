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
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestSimilarityFocusOp(OpTest):

    def setUp(self):
        self.op_type = "similarity_focus"
        batch_size = 2
        x_dim, y_dim, z_dim = 3, 2, 2
        self.inputs = {
            'X':
            np.array([[[[0.8, 0.1], [0.4, 0.5]], [[0.9, 0.7], [0.9, 0.9]],
                       [[0.8, 0.9], [0.1, 0.2]]],
                      [[[0.2, 0.5], [0.3, 0.4]], [[0.9, 0.7], [0.8, 0.4]],
                       [[0.0, 0.2], [0.4, 0.7]]]]),
        }
        self.attrs = {
            'axis': 1,
            'indexes': [0],
        }

        output = None
        for batch in range(batch_size):
            res = np.zeros((1, y_dim, z_dim)).astype("float32").reshape(-1)
            for index in self.attrs['indexes']:
                channel = self.inputs['X'][batch,
                                           index, :, :].reshape(-1).copy()
                tag1 = [0 for i in range(y_dim)]
                tag2 = [0 for i in range(z_dim)]
                cnt = 0
                for i in range(channel.size):
                    index = channel.argmax()
                    idx1 = index // z_dim
                    idx2 = index % z_dim
                    if tag1[idx1] + tag2[idx2] == 0:
                        tag1[idx1] = 1
                        tag2[idx2] = 1
                        res[index] = 1
                        cnt += 1
                        if cnt == min(y_dim, z_dim):
                            break
                    channel[index] = -1
            res = res.reshape(1, y_dim, z_dim).repeat([x_dim], axis=0)
            res = res.reshape(1, x_dim, y_dim, z_dim)
            if output is not None:
                output = np.concatenate((output, res), axis=0)
            else:
                output = res
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


class TestSimilarityFocusOp_axis1(OpTest):

    def setUp(self):
        self.op_type = "similarity_focus"
        batch_size = 3
        x_dim, y_dim, z_dim = 4, 5, 6
        self.inputs = {
            'X':
            np.random.random(
                (batch_size, x_dim, y_dim, z_dim)).astype("float32"),
        }
        self.attrs = {
            'axis': 1,
            'indexes': [0, 3],
        }

        output = None
        for batch in range(batch_size):
            res = np.zeros((1, y_dim, z_dim)).astype("float32").reshape(-1)
            for index in self.attrs['indexes']:
                channel = self.inputs['X'][batch,
                                           index, :, :].reshape(-1).copy()
                tag1 = [0 for i in range(y_dim)]
                tag2 = [0 for i in range(z_dim)]
                cnt = 0
                for i in range(channel.size):
                    index = channel.argmax()
                    idx1 = index // z_dim
                    idx2 = index % z_dim
                    if tag1[idx1] + tag2[idx2] == 0:
                        tag1[idx1] = 1
                        tag2[idx2] = 1
                        res[index] = 1
                        cnt += 1
                        if cnt == min(y_dim, z_dim):
                            break
                    channel[index] = -1
            res = res.reshape(1, y_dim, z_dim)
            res = res.repeat([x_dim], axis=0)
            res = res.reshape(1, x_dim, y_dim, z_dim)
            if output is not None:
                output = np.concatenate((output, res), axis=0)
            else:
                output = res
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


class TestSimilarityFocusOp_axis2(OpTest):

    def setUp(self):
        self.op_type = "similarity_focus"
        batch_size = 6
        x_dim, y_dim, z_dim = 7, 8, 9
        self.inputs = {
            'X':
            np.random.random(
                (batch_size, x_dim, y_dim, z_dim)).astype("float32"),
        }
        self.attrs = {
            'axis': 2,
            'indexes': [0, 3, 5],
        }

        output = None
        for batch in range(batch_size):
            res = np.zeros((x_dim, 1, z_dim)).astype("float32").reshape(-1)
            for index in self.attrs['indexes']:
                channel = self.inputs['X'][batch, :,
                                           index, :].reshape(-1).copy()
                tag1 = [0 for i in range(x_dim)]
                tag2 = [0 for i in range(z_dim)]
                cnt = 0
                for i in range(channel.size):
                    index = channel.argmax()
                    idx1 = index // z_dim
                    idx2 = index % z_dim
                    if tag1[idx1] + tag2[idx2] == 0:
                        tag1[idx1] = 1
                        tag2[idx2] = 1
                        res[index] = 1
                        cnt += 1
                        if cnt == min(x_dim, z_dim):
                            break
                    channel[index] = -1
            res = res.reshape(x_dim, 1, z_dim)
            res = res.repeat([y_dim], axis=1)
            res = res.reshape(1, x_dim, y_dim, z_dim)
            if output is not None:
                output = np.concatenate((output, res), axis=0)
            else:
                output = res
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


class TestSimilarityFocusOp_axis3(OpTest):

    def setUp(self):
        self.op_type = "similarity_focus"
        batch_size = 64
        x_dim, y_dim, z_dim = 48, 48, 13
        self.inputs = {
            'X':
            np.random.random(
                (batch_size, x_dim, y_dim, z_dim)).astype("float32"),
        }
        self.attrs = {
            'axis': 3,
            'indexes': [0, 2, 7, 9],
        }

        output = None
        for batch in range(batch_size):
            res = np.zeros((x_dim, y_dim, 1)).astype("float32").reshape(-1)
            for index in self.attrs['indexes']:
                channel = self.inputs['X'][batch, :, :,
                                           index].reshape(-1).copy()
                tag1 = [0 for i in range(x_dim)]
                tag2 = [0 for i in range(y_dim)]
                cnt = 0
                for i in range(channel.size):
                    index = channel.argmax()
                    idx1 = index // y_dim
                    idx2 = index % y_dim
                    if tag1[idx1] + tag2[idx2] == 0:
                        tag1[idx1] = 1
                        tag2[idx2] = 1
                        res[index] = 1
                        cnt += 1
                        if cnt == min(x_dim, y_dim):
                            break
                    channel[index] = -1
            res = res.reshape(x_dim, y_dim, 1)
            res = res.repeat([z_dim], axis=2)
            res = res.reshape(1, x_dim, y_dim, z_dim)
            if output is not None:
                output = np.concatenate((output, res), axis=0)
            else:
                output = res
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()


class TestSimilarityFocusOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            data = fluid.data(name='data', shape=[16, 3, 2, 2], dtype='float32')

            def test_input_Variable():
                input = np.random.rand(16, 3, 2, 2).astype("float32")
                out = fluid.layers.similarity_focus(input=input,
                                                    axis=1,
                                                    indexes=[0])

            self.assertRaises(TypeError, test_input_Variable)

            def test_axis_Int():
                axis = 1.0
                out = fluid.layers.similarity_focus(input=data,
                                                    axis=axis,
                                                    indexes=[0])

            self.assertRaises(TypeError, test_axis_Int)

            def test_indexes_List():
                indexes = 0
                out = fluid.layers.similarity_focus(input=data,
                                                    axis=1,
                                                    indexes=indexes)

            self.assertRaises(TypeError, test_indexes_List)


if __name__ == "__main__":
    unittest.main()
