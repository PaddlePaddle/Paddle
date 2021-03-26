# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import unittest

import paddle
import numpy as np
import paddle.distributed as dist
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
import paddle.distributed.fleet as fleet

paddle.seed(1024)
np.random.seed(2021)

batch = 5
in_dim = 10
out_dim = 20


class IdentityLayer2D(fluid.dygraph.Layer):
    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight",
            initializer=paddle.nn.initializer.Normal(
                mean=0.0, std=2.0))
        self.weight = self.create_parameter(
            attr=weight_attr, shape=[m, n], dtype='float32', is_bias=False)

    def forward(self):
        return self.weight


class TestDistTraning(unittest.TestCase):
    def test_multiple_gpus(self):
        strategy = fleet.DistributedStrategy()
        model_parallel_size = 8
        strategy.hybrid_configs = {
            "num_data_parallel": 1,
            "num_model_parallel": model_parallel_size,
            "num_pipeline_parallel": 1
        }
        fleet.init(is_collective=True, strategy=strategy)

        # test X * Y
        # input_size_coeff = 13
        # input_size = input_size_coeff * model_parallel_size
        # output_size_coeff = 17
        # output_size = output_size_coeff * model_parallel_size
        # batch_size = 7

        # identity_layer = IdentityLayer2D(batch_size, input_size)
        # linear_layer = fleet.parallel_layer.ParallelLinear(
        #                     size=(input_size, output_size),
        #                     axis=1,
        #                     num_partitions=model_parallel_size)
        # loss_weight = paddle.randn([output_size, 20], 'float32')
        # # forward
        # input_ = identity_layer()
        # output = linear_layer(input_)
        # loss = paddle.matmul(output, loss_weight)

        # print(loss.numpy())
        # loss.sum().backward()

        # weight_attr=paddle.nn.initializer.Normal(
        #         mean=0.0, std=0.01)     

        # optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        # optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

        # dist.init_parallel_env()

        # self.trainer_id = dist.get_rank()

        # model_a = SimpleNet(self.trainer_id)
        # model_b = SimpleNet(self.trainer_id)

        # state_dict = model_a.state_dict()
        # model_b.set_state_dict(state_dict)

        # model_a = paddle.DataParallel(model_a)
        # model_b = paddle.DataParallel(model_b)

        # ones_input = paddle.ones(shape=(batch, in_dim))
        # ones_input.stop_gradient = True

        # w1_grad_sum = np.zeros((in_dim, out_dim), dtype='float32')
        # w2_grad_sum = np.zeros((in_dim, out_dim), dtype='float32')

        # for step_id in range(5):
        #     random_input = paddle.rand(shape=(batch, in_dim))
        #     random_input.stop_gradient = True

    # if step_id % 2 == 0:
    #     out_a = model_a(random_input)
    #     out_b = model_b(random_input)
    # else:
    #     out_a = model_a(ones_input)
    #     out_b = model_b(ones_input)

    # out_a.sum().backward()
    # out_b.sum().backward()

    # self.check_gradient(model_a.parameters())
    # self.check_gradient(model_b.parameters())

    # # test acc gradient
    # w1_grad_sum = self.check_acc(model_a._layers.w1.grad, w1_grad_sum,
    #                              model_b._layers.w1.grad)
    # w2_grad_sum = self.check_acc(model_a._layers.w2.grad, w2_grad_sum,
    #                              model_b._layers.w2.grad)

    # model_a.clear_gradients()

    # def check_acc(self, grad, grad_sum, acc_grad):
    #     if grad is not None:
    #         grad_sum = grad_sum + grad
    #         np.testing.assert_allclose(grad_sum, acc_grad, rtol=1e-6)
    #     return grad_sum

    # def print_trainer_0(self, *args):
    #     if self.trainer_id == 0:
    #         print(*args)

    # def broadcast_param(self, param, root):
    #     paddle.distributed.broadcast(param, root)
    #     return param

    # def check_gradient(self, params):
    #     other_param = []
    #     for param in params:
    #         if param.trainable and (param._grad_ivar() is not None):
    #             grad = param._grad_ivar()
    #             other_grad = self.broadcast_param(grad.clone(), root=1)
    #             if self.trainer_id == 0:
    #                 np.testing.assert_allclose(other_grad.numpy(), grad.numpy())


if __name__ == '__main__':
    unittest.main()
