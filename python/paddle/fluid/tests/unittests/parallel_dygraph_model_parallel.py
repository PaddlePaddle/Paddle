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
import random
import paddle.fluid.generator as generator


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    fleet.mpu.model_parallel_random_seed(seed)


class IdentityLayer2D(fluid.dygraph.Layer):
    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Normal(
                mean=0.0, std=2.0))
        self.weight = self.create_parameter(
            attr=weight_attr,
            shape=[m, n],
            dtype=self._helper.get_default_dtype(),
            is_bias=False)

    def forward(self):
        return self.weight


class SplitLayer2D(fluid.dygraph.Layer):
    def __init__(self, m, n, rank, mp_word_size):
        super(SplitLayer2D, self).__init__()
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Normal(
                mean=0.0, std=2.0))
        self.master_weight = self.create_parameter(
            attr=weight_attr,
            shape=[m, n],
            dtype=self._helper.get_default_dtype(),
            is_bias=False)
        self.rank = rank
        self.mp_word_size = mp_word_size

    def forward(self):
        self.all_weight = paddle.split(
            self.master_weight, num_or_sections=self.mp_word_size, axis=1)

        return self.all_weight[self.rank], self.master_weight


class TestDistTraning(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        strategy.hybrid_configs = {
            "num_data_parallel": 1,
            "num_model_parallel": self.model_parallel_size,
            "num_pipeline_parallel": 1
        }

        strategy.model_parallel_configs = {"global_seed": 1024}
        fleet.init(is_collective=True, strategy=strategy)

    # def test_column_parallel_layer(self):
    #     set_random_seed(1024)
    #     global_dtype = "float32"
    #     paddle.set_default_dtype(global_dtype)

    #     input_size_coeff = 13
    #     input_size = input_size_coeff * self.model_parallel_size
    #     output_size_coeff = 17
    #     output_size = output_size_coeff * self.model_parallel_size
    #     batch_size = 7

    #     identity_layer = IdentityLayer2D(batch_size, input_size)
    #     linear_layer = fleet.mpu.ColumnParallelLinear(
    #                             in_features=input_size,
    #                             out_features=output_size,
    #                             weight_attr=None,
    #                             has_bias=True,
    #                             gather_output=True,
    #                             name="test_column_linear")
    #     loss_weight = paddle.randn([output_size, 20], global_dtype)

    #     # forward
    #     input_ = identity_layer()
    #     output = linear_layer(input_)
    #     loss = paddle.matmul(output, loss_weight).sum()

    #     # backward
    #     loss.backward()

    #     # check values just for test
    #     check_group = dist.new_group(list(range(self.model_parallel_size)))
    #     integral_W = []
    #     partial_W = linear_layer.weight.clone().detach()
    #     paddle.distributed.all_gather(
    #         integral_W, partial_W, group=check_group)
    #     integral_W = paddle.concat(integral_W, axis=len(partial_W.shape) - 1)

    #     integral_B = []
    #     partial_B = linear_layer.bias.clone().detach()
    #     paddle.distributed.all_gather(
    #         integral_B, partial_B, group=check_group)
    #     integral_B = paddle.concat(integral_B, axis=len(partial_B.shape) - 1)

    #     actual_output = paddle.matmul(input_, integral_W) + integral_B
    #     actual_loss = paddle.matmul(actual_output, loss_weight).sum()
    #     actual_loss.backward()

    #     np.testing.assert_allclose(output.numpy(), actual_output.numpy())
    #     np.testing.assert_allclose(output.grad, actual_output.grad)

    def test_row_parallel_layer(self):
        global_dtype = "float32"
        paddle.set_default_dtype(global_dtype)

        self.hcg = fleet.get_hybrid_communicate_group()

        self.word_size = self.hcg.get_model_parallel_world_size()
        self.rank_id = self.hcg.get_model_parallel_rank()

        input_size_coeff = 2
        input_size = input_size_coeff * self.model_parallel_size
        output_size_coeff = 2
        output_size = output_size_coeff * self.model_parallel_size
        batch_size = 2

        identity_layer = IdentityLayer2D(batch_size, input_size)

        linear_layer = fleet.mpu.RowParallelLinear(
            in_features=input_size,
            out_features=output_size,
            has_bias=True,
            input_is_parallel=False)

        check_group = dist.new_group(list(range(self.model_parallel_size)))
        # check values just for test
        integral_W = []
        partial_W = linear_layer.weight.clone().detach()
        paddle.distributed.all_gather(integral_W, partial_W, group=check_group)
        integral_W = paddle.concat(integral_W, axis=0)

        # integral_B = paddle.create_parameter()
        integral_ww = paddle.create_parameter(
            shape=[input_size, output_size],
            dtype=global_dtype,
            attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(integral_W)))
        integral_ww.trainable = True

        integral_bb = paddle.create_parameter(
            shape=[output_size],
            dtype=global_dtype,
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)))
        integral_bb.trainable = True

        optimizer_a = paddle.optimizer.SGD(learning_rate=0.001,
                                           parameters=linear_layer.parameters())

        optimizer_b = paddle.optimizer.SGD(
            learning_rate=0.001, parameters=[integral_ww, integral_bb])

        for idx in range(5):
            input_ = identity_layer()

            # forward
            output = linear_layer(input_.detach())
            loss = output.sum()

            actual_output = paddle.matmul(input_.detach(),
                                          integral_ww) + integral_bb
            actual_loss = actual_output.sum()

            # np.testing.assert_allclose(output.numpy(), actual_output.numpy())
            # backward
            meg_loss = loss / 2
            meg_loss.backward()
            actual_loss.backward()
            if dist.get_rank() == 0:
                # print("ids :",idx)
                # print(output.numpy()- actual_output.numpy())
                print('w grad {}  : {}'.format(integral_ww.grad,
                                               linear_layer.weight.grad))
            # optimizer_a.step()
            # optimizer_b.step()

            # optimizer_a.clear_grad()
            # optimizer_b.clear_grad()

            # def test_parallel_embedding(self):
            #     batch_size = 17
            #     seq_length = 23
            #     vocab_size_per_card = 2
            #     vocab_size = vocab_size_per_card *  self.model_parallel_size
            #     hidden_size = 2
            #     seed = 1236

            #     set_random_seed(123)
            #     loss_weight = paddle.randn([batch_size, seq_length, hidden_size])
            #     rank_id = dist.get_rank()

            #     # model_a
            #     embedding_vocab_parallel = fleet.mpu.VocabParallelEmbedding(
            #                                  vocab_size, hidden_size)
            #     check_group = dist.new_group(list(range(self.model_parallel_size)))

            #     # model_b
            #     integral_W = []
            #     partial_W = embedding_vocab_parallel.embedding.weight.clone().detach()
            #     paddle.distributed.all_gather(
            #         integral_W, partial_W, group=check_group)
            #     # just for test, we delete the last dim
            #     result_w = []
            #     for idx in range(len(integral_W)):
            #         tmp = paddle.gather(integral_W[idx], paddle.to_tensor(list(range(vocab_size_per_card))))
            #         result_w.append(tmp)
            #     # result_w.append(integral_W[-1])
            #     integral_W = paddle.concat(result_w, axis=0)
            #     origin_weight_attr = paddle.framework.ParamAttr(
            #             name="origin_embedding",
            #             initializer=paddle.nn.initializer.Assign(integral_W))
            #     embedding_original = paddle.nn.Embedding(vocab_size, hidden_size,
            #                                              weight_attr=origin_weight_attr,
            #                                              )

            #     optimizer_a = paddle.optimizer.SGD(learning_rate=0.001,
            #                                     parameters=embedding_vocab_parallel.parameters())

            #     optimizer_b = paddle.optimizer.SGD(learning_rate=0.001,
            #                                         parameters=embedding_original.parameters())
            #     for _ in range(5):
            #         # np_input_data = np.random.randint(0, vocab_size, (batch_size, seq_length))
            #         np_input_data = np.array([[0,0],
            #                                   [1,1]])
            #         input_data = paddle.to_tensor(np_input_data, dtype="int32")

            #         output_a = embedding_vocab_parallel(input_data)
            #         loss_a = output_a.sum()

            #         output_b = embedding_original(input_data)
            #         loss_b = output_b.sum()

            #         np.testing.assert_allclose(loss_a.numpy(), loss_b.numpy())

            #         loss_a.backward()
            #         loss_b.backward()

            #         # print("output_a grad {}  output_b grad {}".format(output_a.grad, output_b.grad))
            #         # if rank_id == 1:
            #         #     print("a  {}, b  {}".format(embedding_vocab_parallel.embedding.weight.numpy(),
            #         #                                     embedding_original.weight.numpy()))
            #         #     print("a grad {}, b grad {}".format(embedding_vocab_parallel.embedding.weight.grad,
            #         #                                     embedding_original.weight.grad))

            #         # optimizer_a.step()
            #         # optimizer_b.step()

            #         # optimizer_a.clear_grad()
            #         # optimizer_b.clear_grad()


if __name__ == '__main__':
    unittest.main()
