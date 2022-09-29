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

import numpy as np
import unittest
import paddle
import paddle.fluid as fluid
import paddle.fluid.initializer as initializer
from paddle.fluid import Program, program_guard

from op_test import OpTest


def nce(input, weight, bias, sample_weight, labels, num_classes,
        num_sample_class):
    samples = []
    sample_labels = []
    batch_size = input.shape[0]
    num_true_class = labels.shape[1]
    for i in range(batch_size):
        w = 1 if sample_weight is None else sample_weight[i]
        for label in labels[i]:
            samples.append((i, label, True, w))
            sample_labels.append(label)
        for num in range(num_sample_class):
            samples.append((i, num, False, w))
            sample_labels.append(num)
    # forward bias
    sample_out = np.zeros(len(samples)).astype(np.float32)
    if bias is not None:
        for i in range(len(samples)):
            sample_out[i] = bias[samples[i][1]]
    # forward weight
    for i in range(len(samples)):
        sample_out[i] += np.dot(input[samples[i][0]], weight[samples[i][1]])

    # forward activation
    sample_out = 1.0 / (1.0 + np.exp(-sample_out))
    # forward cost
    out = np.zeros(batch_size).astype(np.float32)
    b = 1.0 / num_classes * num_sample_class
    for i in range(len(samples)):
        o = sample_out[i]
        cost = -np.log(o / (o + b)) if samples[i][2] else -np.log(b / (o + b))
        out[samples[i][0]] += cost * samples[i][3]
    return (out[:, np.newaxis],
            np.array(sample_out).reshape(batch_size,
                                         num_sample_class + num_true_class),
            np.array(sample_labels).reshape(batch_size,
                                            num_sample_class + num_true_class))


class TestNCE(OpTest):

    def generate_data(self, dim, batch_size, num_classes, num_true_class,
                      num_neg_samples, is_sparse):
        input = np.random.randn(batch_size, dim).astype(np.float32)
        weight = np.random.randn(num_classes, dim).astype(np.float32)
        bias = np.random.randn(num_classes).astype(np.float32)
        sample_weight = np.random.randn(batch_size).astype(np.float32)
        labels = np.random.randint(0, num_classes,
                                   (batch_size, num_true_class)).astype("int64")
        self.attrs = {
            'num_total_classes': num_classes,
            'num_neg_samples': num_neg_samples,
            'custom_neg_classes': list(range(num_neg_samples)),
            'seed': 0,
            'sampler': 0,
            'is_sparse': is_sparse,
            'is_test': self.is_test
        }
        self.inputs = {
            'Input': input,
            'Label': labels,
            'Weight': weight,
            'Bias': bias,
            'SampleWeight': sample_weight
        }

    def set_is_test(self):
        self.is_test = False

    def set_data(self):
        self.generate_data(5, 25, 100, 1, 2, False)

    def compute(self):
        out = nce(self.inputs['Input'], self.inputs['Weight'],
                  self.inputs['Bias'], self.inputs['SampleWeight'],
                  self.inputs['Label'], self.attrs['num_total_classes'],
                  self.attrs['num_neg_samples'])
        if self.is_test:
            self.outputs = {'Cost': out[0]}
        else:
            self.outputs = {
                'Cost': out[0],
                'SampleLogits': out[1],
                'SampleLabels': out[2]
            }

    def setUp(self):
        self.op_type = 'nce'
        self.set_is_test()
        self.set_data()
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["Input", "Weight", "Bias"],
                        "Cost",
                        max_relative_error=0.02)


class TestNCECase1Tensor(TestNCE):

    def set_data(self):
        self.generate_data(10, 20, 100, 2, 5, False)


class TestNCETensorIsTest(TestNCE):
    # if is_test = True, there's no need to calculate grad
    def set_is_test(self):
        self.is_test = True

    def test_check_grad(self):
        pass


class TestNCECase1SelectedRows(unittest.TestCase):

    def setUp(self):
        self.base_lr = 0.0001
        self.batch_size = 8

    @staticmethod
    def get_place():
        place = fluid.core.CPUPlace()
        return place

    @staticmethod
    def get_train_data(batch_size):
        batches = []
        for i in range(batch_size):
            input = np.random.randn(batch_size, 10).astype(np.float32)
            labels = np.random.randint(0, 20, (batch_size, 1))
            batches.append([input, labels])
        return batches

    def get_optimizer(self):
        # SGD optimizer
        optimizer = fluid.optimizer.SGD(learning_rate=self.base_lr)
        return optimizer

    def train_network(self, num_total_classes, num_neg_samples, sampler,
                      custom_dist, is_sparse):
        input = fluid.layers.data(name="input", shape=[10], dtype="float32")
        label = fluid.layers.data(name="label", shape=[1], dtype="int64")

        w_param = fluid.default_main_program().global_block().create_parameter(
            shape=[num_total_classes, 10],
            dtype='float32',
            name='nce_w',
            initializer=initializer.ConstantInitializer())
        b_param = fluid.default_main_program().global_block().create_parameter(
            shape=[num_total_classes, 1],
            dtype='float32',
            name='nce_b',
            initializer=initializer.ConstantInitializer())

        cost = fluid.layers.nce(input=input,
                                label=label,
                                num_total_classes=num_total_classes,
                                sampler=sampler,
                                custom_dist=custom_dist,
                                sample_weight=None,
                                param_attr='nce_w',
                                bias_attr='nce_b',
                                seed=1,
                                num_neg_samples=num_neg_samples,
                                is_sparse=is_sparse)
        avg_cost = paddle.mean(cost)
        # optimizer
        optimizer = self.get_optimizer()
        optimizer.minimize(avg_cost)

        return [avg_cost, [input, label]]

    def test_input_is_selected_rows(self):
        place = self.get_place()
        exe = fluid.Executor(place)

        data = self.get_train_data(self.batch_size)
        nid_freq_arr = np.random.dirichlet(np.ones(20) * 1000).astype('float32')

        rets = []
        # for dense
        dense_scope = fluid.core.Scope()
        dense_startup_program = fluid.framework.Program()
        dense_train_program = fluid.framework.Program()
        with fluid.scope_guard(dense_scope):
            with fluid.program_guard(dense_train_program,
                                     dense_startup_program):
                cost, feeds = self.train_network(20, 5, "custom_dist",
                                                 nid_freq_arr.tolist(), False)
                feeder = fluid.DataFeeder(feed_list=feeds, place=place)
                exe.run(dense_startup_program)
                loss_val = exe.run(dense_train_program,
                                   feed=feeder.feed(data),
                                   fetch_list=[cost.name])
                rets.append(np.mean(loss_val))

        # for sparse
        sparse_scope = fluid.core.Scope()
        sparse_startup_program = fluid.framework.Program()
        sparse_train_program = fluid.framework.Program()
        with fluid.scope_guard(sparse_scope):
            with fluid.program_guard(sparse_train_program,
                                     sparse_startup_program):
                cost, feeds = self.train_network(20, 5, "custom_dist",
                                                 nid_freq_arr.tolist(), True)
                feeder = fluid.DataFeeder(feed_list=feeds, place=place)
                exe.run(sparse_startup_program)
                loss_val = exe.run(sparse_train_program,
                                   feed=feeder.feed(data),
                                   fetch_list=[cost.name])
                rets.append(np.mean(loss_val))

        self.assertEqual(rets[0], rets[1])


class TestNCE_OpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            input1 = fluid.create_lod_tensor(np.array([0.0, 3.0, 2.0, 4.0]),
                                             [[1, 1, 2]], fluid.CPUPlace())
            label1 = fluid.layers.data(name='label1',
                                       shape=[-1, 4],
                                       dtype="int64")
            # the input(input) of nce layer must be Variable.
            self.assertRaises(TypeError, fluid.layers.nce, input1, label1, 5)

            input2 = fluid.layers.data(name='input2',
                                       shape=[-1, 4],
                                       dtype="float32")
            label2 = fluid.create_lod_tensor(np.array([0.0, 3.0, 2.0, 4.0]),
                                             [[1, 1, 2]], fluid.CPUPlace())
            # the input(label) of nce layer must be Variable.
            self.assertRaises(TypeError, fluid.layers.nce, input2, label2, 5)

            input3 = fluid.layers.data(name='input3',
                                       shape=[-1, 4],
                                       dtype="float16")
            label3 = fluid.layers.data(name='label3',
                                       shape=[-1, 1],
                                       dtype="int64")
            # the data type of input(input) must be float32 or float64.
            self.assertRaises(TypeError, fluid.layers.nce, input3, label3, 5)

            input4 = fluid.layers.data(name='input4',
                                       shape=[-1, 4],
                                       dtype="float32")
            label4 = fluid.layers.data(name='label4',
                                       shape=[-1, 1],
                                       dtype="int32")
            # the data type of input(label) must be int64.
            self.assertRaises(TypeError, fluid.layers.nce, input4, label4, 5)


class TestDygraphNCE_OpError(unittest.TestCase):

    def test_NCE_errors(self):
        with program_guard(Program(), Program()):
            nce = fluid.NCE(20, 5)
            input1 = fluid.create_lod_tensor(np.array([0.0, 3.0, 2.0, 4.0]),
                                             [[1, 1, 2]], fluid.CPUPlace())
            label1 = fluid.layers.data(name='label1',
                                       shape=[-1, 4],
                                       dtype="int64")
            # the input(input) of NCE layer must be Variable.
            self.assertRaises(TypeError, nce, input1, label1)

            input2 = fluid.layers.data(name='input2',
                                       shape=[-1, 4],
                                       dtype="float32")
            label2 = fluid.create_lod_tensor(np.array([0.0, 3.0, 2.0, 4.0]),
                                             [[1, 1, 2]], fluid.CPUPlace())
            # the input(label) of NCE layer must be Variable.
            self.assertRaises(TypeError, nce, input2, label2)

            input3 = fluid.layers.data(name='input3',
                                       shape=[-1, 4],
                                       dtype="float16")
            label3 = fluid.layers.data(name='label3',
                                       shape=[-1, 1],
                                       dtype="int64")
            # the data type of input(input) must be float32 or float64.
            self.assertRaises(TypeError, nce, input3, label3)

            input4 = fluid.layers.data(name='input4',
                                       shape=[-1, 4],
                                       dtype="float32")
            label4 = fluid.layers.data(name='label4',
                                       shape=[-1, 1],
                                       dtype="int32")
            # the data type of input(label) must be int64.
            self.assertRaises(TypeError, nce, input4, label4)

            input5 = fluid.layers.data(name='input5',
                                       shape=[-1, 4],
                                       dtype="float32")
            label5 = fluid.layers.data(name='label5',
                                       shape=[-1, 1],
                                       dtype="int64")
            sample_weight = fluid.create_lod_tensor(
                np.array([0.0, 3.0, 2.0, 4.0]), [[1, 1, 2]], fluid.CPUPlace())
            # the sample_weight of nce must be Variable or None.
            self.assertRaises(TypeError, nce, input5, label5, sample_weight)


if __name__ == '__main__':
    unittest.main()
