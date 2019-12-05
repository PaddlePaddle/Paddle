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
import collections
from simple_nets import init_data


class Simple_Net():
    def __init__(self):
        self.stop_gradient_vars = []
        self.no_grad_set = []

    def build_model(self):
        # stop_gradient = True in input
        x = fluid.data(name='x_no_grad', shape=[13], dtype='int64')
        x2 = fluid.data(name='x2_no_grad', shape=[13], dtype='int64')
        x3 = fluid.data(name='x3_no_grad', shape=[13], dtype='int64')
        label = fluid.data(name='label_no_grad', shape=[13, 1], dtype='int64')
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

    def check_params_grad(self, loss, parameter_list=None, no_grad_set=None):
        params_grads = fluid.backward.append_backward(loss, parameter_list,
                                                      no_grad_set)
        gt = [u'w2v', u'fc_predict.b_0', u'fc_w']
        params_names = set([var[0].name for var in params_grads])
        assert len(params_names - set(gt)) == 0

        return params_grads

    def check_stop_gradient(self, program):
        no_grad_dict = fluid.backward._get_stop_gradients_(program)
        gt = [
            u'x_no_grad@GRAD', u'x2_no_grad@GRAD', u'x3_no_grad@GRAD',
            u'label_no_grad@GRAD'
        ]
        if no_grad_dict is not None and isinstance(no_grad_dict, dict):
            assert len(no_grad_dict[0] - set(gt)) == 0

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
        gt = [
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
        assert op_types == gt

        return op_path, block_no_grad_set

    def check_find_no_grad_vars(self, root_block, op_path, targets,
                                block_no_grad_set):
        no_grad_vars = fluid.backward._find_no_grad_vars(
            root_block, op_path, targets, block_no_grad_set)
        assert len(no_grad_vars) == 0

        return no_grad_vars


def case2_prune_no_grad_branch():
    x = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feature = fluid.layers.fc(input=x, size=10, act=None)
    label = fluid.layers.cast(label, dtype="float32")
    label = fluid.layers.cast(label, dtype='int64')
    # Note that the label is not persistable in fluid.layers.cross_entropy.
    loss = fluid.layers.cross_entropy(input=feature, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def case3_prune_no_grad_branch2():
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    label = fluid.layers.cast(label, dtype="float32")
    label = fluid.layers.cast(label, dtype='int64')
    out = fluid.layers.one_hot(input=label, depth=100)
    loss = fluid.layers.mean(out)
    return loss


def case4_with_no_grad_op_maker():
    out = fluid.layers.gaussian_random(shape=[20, 30])
    loss = fluid.layers.mean(out)
    return loss


class TestBackward(unittest.TestCase):
    def check_backward(self, net, feed_dict):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        main = fluid.Program()
        startup = fluid.Program()

        with fluid.program_guard(main, startup):
            loss = net.build_model()

            optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            # test each step of minimize
            # step 1: append_backward

            params_grads = net.check_params_grad(loss)
            print(params_grads)
            # 1.1 get_stop_gradients
            no_grad_dict = net.check_stop_gradient(main)
            print(no_grad_dict)
            # 1.2 find_op_path
            op_path, block_no_grad_set = net.check_op_path(
                main.block(0), [loss], [], no_grad_dict)
            print(op_path)
            # 1.3 _find_no_grad_vars
            no_grad_vars = net.check_find_no_grad_vars(
                main.block(0), op_path, [loss], block_no_grad_set)
            print(no_grad_vars)
            #
            block_no_grad_set.update(no_grad_vars)
            no_grad_dict[0].update(
                list(
                    map(fluid.backward._append_grad_suffix_,
                        block_no_grad_set)))

            # run whole net
            optimizer.minimize(loss)
            # exe.run(feed=feed_dict)

    def test_backward(self):
        # batch_size = 2
        # img, label = init_data(batch_size, img_shape=[784], label_range=9)
        # feed_dict = {'image': img, 'label': label}
        simple_net = Simple_Net()
        self.check_backward(simple_net, {})
        # self.check_backward(case2_prune_no_grad_branch, feed_dict)
        # self.check_backward(case3_prune_no_grad_branch2, {'label': label})
        # self.check_backward(case4_with_no_grad_op_maker, {})


if __name__ == '__main__':
    unittest.main()
