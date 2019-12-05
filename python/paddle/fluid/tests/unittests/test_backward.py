# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle.fluid as fluid
import numpy as np


class BackwardNet(object):
    def __init__(self):
        self.stop_gradient_grad_vars = []
        self.no_grad_vars = []
        self.params_names = []
        self.op_path = []

    def build_model(self):
        raise NotImplementedError

    def init_data(self):
        raise NotImplementedError

    def check_backward(self, loss, main_program):
        params_grads = self.check_params_grad(loss)
        # 1.1 get_stop_gradients
        no_grad_dict = self.check_stop_gradient(main_program)
        # 1.2 find_op_path
        op_path, block_no_grad_set = self.check_op_path(
            main_program.block(0), [loss], [], no_grad_dict)
        # 1.3 _find_no_grad_vars
        no_grad_vars = self.check_find_no_grad_vars(
            main_program.block(0), op_path, [loss], block_no_grad_set)
        # update no_grad_dict
        block_no_grad_set.update(no_grad_vars)
        no_grad_dict[0].update(
            list(map(fluid.backward._append_grad_suffix_, block_no_grad_set)))

    def check_params_grad(self, loss, parameter_list=None, no_grad_set=None):
        params_grads = fluid.backward.append_backward(loss, parameter_list,
                                                      no_grad_set)
        params_names = set([var[0].name for var in params_grads])
        assert len(params_names - set(self.params_names)) == 0

        return params_grads

    def check_stop_gradient(self, program):
        no_grad_dict = fluid.backward._get_stop_gradients_(program)
        if no_grad_dict is not None and isinstance(no_grad_dict, dict):
            assert len(no_grad_dict[0] - set(self.stop_gradient_grad_vars)) == 0

        return no_grad_dict

    def check_op_path(self, root_block, outputs, inputs=[], no_grad_dict=None):
        if no_grad_dict is None or not isinstance(no_grad_dict, dict):
            block_no_grad_set = None
        else:
            block_no_grad_set = set(
                map(fluid.backward._strip_grad_suffix_, no_grad_dict[0]))
        op_path = fluid.backward._find_op_path_(root_block, outputs, inputs,
                                                block_no_grad_set)
        op_types = [op.type for op in op_path]
        assert op_types == self.op_path

        return op_path, block_no_grad_set

    def check_find_no_grad_vars(self, root_block, op_path, targets,
                                block_no_grad_set):
        no_grad_vars = fluid.backward._find_no_grad_vars(
            root_block, op_path, targets, block_no_grad_set)
        assert len(no_grad_vars) == len(self.no_grad_vars)

        return no_grad_vars


class SimpleNet(BackwardNet):
    def __init__(self):
        super(BackwardNet, self).__init__()
        self.stop_gradient_grad_vars = [
            u'x_no_grad@GRAD', u'x2_no_grad@GRAD', u'x3_no_grad@GRAD',
            u'label_no_grad@GRAD'
        ]
        self.no_grad_vars = []
        self.params_names = [u'w2v', u'fc_predict.b_0', u'fc_w']
        self.op_path = [
            u'lookup_table_v2',
            u'lookup_table_v2',  # embedding
            u'elementwise_add',  # merge
            u'mul',
            u'elementwise_add',
            u'softmax',  # fc
            u'elementwise_sub',
            u'square',
            u'mean'
        ]  # loss
        self.shape = [16, 50]

    def init_data(self):
        assert len(self.shape) == 2
        x = np.random.randint(0, 90, self.shape).astype('int64')
        x2 = np.random.randint(0, 90, self.shape).astype('int64')
        x3 = np.random.randint(0, 90, self.shape).astype('int64')
        label = np.random.random([self.shape[0], 1]).astype('float32')
        return {
            'x_no_grad': x,
            'x2_no_grad': x2,
            'x3_no_grad': x3,
            'label_no_grad': label
        }

    def build_model(self):
        # stop_gradient = True in input
        x = fluid.data(name='x_no_grad', shape=self.shape, dtype='int64')
        x2 = fluid.data(name='x2_no_grad', shape=self.shape, dtype='int64')
        x3 = fluid.data(name='x3_no_grad', shape=self.shape, dtype='int64')
        label = fluid.data(
            name='label_no_grad', shape=[self.shape[0], 1], dtype='float32')
        # shared layer, the grad of 'w2v' will be summed and renamed.
        # To test  _addup_repetitive_outputs_
        x_emb = fluid.embedding(
            x, size=[100, 64], param_attr=fluid.ParamAttr(name='w2v'))
        x2_emb = fluid.embedding(
            x2, size=[100, 64], param_attr=fluid.ParamAttr(name='w2v'))
        x3_emb = fluid.embedding(
            x3, size=[100, 64], param_attr=fluid.ParamAttr(name='w2v'))
        # merge layers
        x_merge = fluid.layers.elementwise_add(x_emb, x2_emb, name='x_add_x2')
        x2_merge = fluid.layers.elementwise_add(
            x2_emb, x3_emb, name='x2_add_x3')
        # shared fc_w
        predict = fluid.layers.fc(input=x_merge,
                                  size=1,
                                  act='softmax',
                                  param_attr=fluid.ParamAttr(name='fc_w'),
                                  name='fc_predict')
        # useless layer for calculating loss
        fc_no_use = fluid.layers.fc(input=x2_merge,
                                    size=1,
                                    act='sigmoid',
                                    param_attr=fluid.ParamAttr(name='fc_w'),
                                    name='fc_no_use')
        # loss
        cost = fluid.layers.square_error_cost(input=predict, label=label)
        loss = fluid.layers.mean(cost, name='mean_loss')

        return loss


# NotImplemented
class ConditionalNet(BackwardNet):
    def __init__(self):
        super(BackwardNet, self).__init__()


class TestBackward(unittest.TestCase):
    def check_backward(self, net):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        main = fluid.Program()
        startup = fluid.Program()

        with fluid.program_guard(main, startup):
            loss = net.build_model()
            net.check_backward(loss, main)

            optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            optimizer.minimize(loss)
            exe.run(startup)
            exe.run(feed=net.init_data())

    def test_backward(self):
        simple_net = SimpleNet()
        self.check_backward(simple_net)


if __name__ == '__main__':
    unittest.main()
