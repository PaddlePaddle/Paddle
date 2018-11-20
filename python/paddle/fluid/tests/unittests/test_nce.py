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

from __future__ import print_function

import numpy as np
import math
import unittest

from op_test import OpTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.framework import Program


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
    return (out[:, np.newaxis], np.array(sample_out).reshape(
        batch_size, num_sample_class + num_true_class),
            np.array(sample_labels).reshape(batch_size,
                                            num_sample_class + num_true_class))


#class TestNCE(OpTest):
#    def generate_data(self, dim, batch_size, num_classes, num_true_class,
#                      num_neg_samples, is_sparse):
#        input = np.random.randn(batch_size, dim).astype(np.float32)
#        weight = np.random.randn(num_classes, dim).astype(np.float32)
#        bias = np.random.randn(num_classes).astype(np.float32)
#        sample_weight = np.random.randn(batch_size).astype(np.float32)
#        labels = np.random.randint(0, num_classes, (batch_size, num_true_class))
#        self.attrs = {
#            'num_total_classes': num_classes,
#            'num_neg_samples': num_neg_samples,
#            'custom_neg_classes': list(range(num_neg_samples)),
#            'seed': 0,
#            'sampler': 0,
#            'is_sparse': is_sparse
#        }
#        self.inputs = {
#            'Input': input,
#            'Label': labels,
#            'Weight': weight,
#            'Bias': bias,
#            'SampleWeight': sample_weight
#        }
#
#    def set_data(self):
#        self.generate_data(5, 5, 4, 1, 2, False)
#
#    def compute(self):
#        out = nce(self.inputs['Input'], self.inputs['Weight'],
#                  self.inputs['Bias'], self.inputs['SampleWeight'],
#                  self.inputs['Label'], self.attrs['num_total_classes'],
#                  self.attrs['num_neg_samples'])
#        self.outputs = {
#            'Cost': out[0],
#            'SampleLogits': out[1],
#            'SampleLabels': out[2]
#        }
#
#    def setUp(self):
#        self.op_type = 'nce'
#        self.set_data()
#        self.compute()
#
#    def test_check_output(self):
#        self.check_output()
#
#    def test_check_grad(self):
#        self.check_grad(
#            ["Input", "Weight", "Bias"], "Cost", max_relative_error=0.02)

#class TestNCECase1Tensor(TestNCE):
#    def set_data(self):
#        self.generate_data(10, 20, 10, 2, 5, False)


class TestNCECase1SelectedRows(unittest.TestCase):
    def setUp(self):
        self.base_lr = 0.0001
        self.batch_size = 8

    @staticmethod
    def get_place():
        place = fluid.core.CPUPlace()
        return place

    @staticmethod
    def get_train_reader(batch_size):
        def reader():
            for x in range(1000):
                batchs = []
                for i in range(batch_size):
                    input = np.random.randint(0, 20, (batch_size, 1))
                    labels = np.random.randint(0, 20, (batch_size, 1))
                    batchs.append([input, labels])
                yield batchs

        return reader

    def get_optimizer(self):
        # SGD optimizer
        optimizer = fluid.optimizer.SGD(learning_rate=self.base_lr)
        return optimizer

    def train_network(self, dict_size, word_frequencys, embedding_size):
        input_word = fluid.layers.data(
            name="input_word", shape=[1], dtype='int64')
        predict_word = fluid.layers.data(
            name='predict_word', shape=[1], dtype='int64')

        emb = fluid.layers.embedding(
            input=input_word,
            size=[dict_size, embedding_size],
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(dict_size))))

        num_total_classes = dict_size
        num_neg_samples = 5
        sampler = "uniform"
        sample_weight = None

        w_param_name = "nce_w"
        b_param_name = "nce_b"

        w_param = fluid.default_main_program().global_block().create_parameter(
            shape=[num_total_classes, embedding_size],
            dtype='float32',
            name=w_param_name)
        b_param = fluid.default_main_program().global_block().create_parameter(
            shape=[num_total_classes, 1], dtype='float32', name=b_param_name)

        cost = fluid.layers.nce(input=emb,
                                label=predict_word,
                                num_total_classes=num_total_classes,
                                sampler=sampler,
                                custom_dist=word_frequencys,
                                sample_weight=sample_weight,
                                param_attr=fluid.ParamAttr(name=w_param_name),
                                bias_attr=fluid.ParamAttr(name=b_param_name),
                                num_neg_samples=num_neg_samples)

        avg_cost = fluid.layers.mean(cost)
        # optimizer
        optimizer = self.get_optimizer()
        optimizer.minimize(avg_cost)
        data_list = [input_word, predict_word]
        return [avg_cost, data_list]

    def test_input_is_selected_rows(self):

        dict_size = 20
        embedding_size = 10
        nid_freq_arr = np.random.dirichlet(np.ones(dict_size) *
                                           1000).astype('float32')

        cost, datas = self.train_network(dict_size, nid_freq_arr,
                                         embedding_size)

        place = self.get_place()
        exe = fluid.Executor(place)

        feeder = fluid.DataFeeder(feed_list=datas, place=place)
        train_reader = self.get_train_reader(self.batch_size)

        startup = fluid.default_startup_program()
        prog = fluid.default_main_program()

        exe.run(startup)
        for batch_id, data in enumerate(train_reader()):
            loss_val = exe.run(prog, feed=feeder.feed(data), fetch_list=[cost])
            print(loss_val)


if __name__ == '__main__':
    unittest.main()
