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

import os
import unittest

import paddle
from paddle.distributed import fleet

paddle.enable_static()


class ColumnLinearNet(paddle.nn.Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.parallel_linear = fleet.meta_parallel.ColumnParallelLinear(
            in_features=input_size,
            out_features=output_size,
            weight_attr=None,
            has_bias=True,
            gather_output=True,
            name="test_column_linear",
        )

    def forward(self, x):
        output = self.parallel_linear(x)
        return output


class RowLinearNet(paddle.nn.Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.parallel_linear = fleet.meta_parallel.RowParallelLinear(
            in_features=input_size,
            out_features=output_size,
            has_bias=True,
            input_is_parallel=False,
            name="test_row_linear",
        )

    def forward(self, x):
        output = self.parallel_linear(x)
        return output


class EmbeddingNet(paddle.nn.Layer):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size, hidden_size
        )

    def forward(self, x):
        output = self.embedding(x)
        return output


class TestDistTraining(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "2"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = (
            "127.0.0.1:36001,127.0.0.1:36002,127.0.0.1:36003,127.0.0.1:36004"
        )

        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        strategy.sharding = True
        strategy.sharding_configs = {
            "mp_degree": self.model_parallel_size,
            "sharding_degree": 2,
        }
        strategy.tensor_parallel = True
        strategy.tensor_parallel_configs = {"tensor_parallel_degree": 2}
        fleet.init(is_collective=True, strategy=strategy)

    def get_program(self):
        return paddle.static.Program(), paddle.static.Program()

    def test_column_parallel_layer(self):
        main_program, startup_program = self.get_program()
        with paddle.static.program_guard(main_program, startup_program):
            input_size, output_size = 28, 64
            model_a = ColumnLinearNet(input_size, output_size)

            x = paddle.static.data(name='x', shape=[None, input_size])
            y = model_a(x)

            # print(main_program)
            ops = main_program.global_block().ops
            ops = [op.type for op in ops]
            self.assertEqual(
                ops, ['c_identity', 'matmul_v2', 'elementwise_add', 'c_concat']
            )

            weight = model_a.parallel_linear.weight
            bias = model_a.parallel_linear.bias
            self.assertEqual(
                weight.shape,
                (input_size, output_size // self.model_parallel_size),
            )
            self.assertEqual(
                bias.shape, (output_size // self.model_parallel_size,)
            )

    def test_row_parallel_layer(self):
        main_program, startup_program = self.get_program()
        with paddle.static.program_guard(main_program, startup_program):
            input_size, output_size = 28, 64
            model_a = RowLinearNet(input_size, output_size)

            x = paddle.static.data(name='x', shape=[None, input_size])
            y = model_a(x)

            # print(main_program)
            ops = main_program.global_block().ops
            ops = [op.type for op in ops]
            self.assertEqual(
                ops,
                ['c_split', 'matmul_v2', 'mp_allreduce_sum', 'elementwise_add'],
            )

            weight = model_a.parallel_linear.weight
            bias = model_a.parallel_linear.bias
            self.assertEqual(
                weight.shape,
                (input_size // self.model_parallel_size, output_size),
            )
            self.assertEqual(bias.shape, (output_size,))

    def test_parallel_embedding(self):
        main_program, startup_program = self.get_program()
        with paddle.static.program_guard(main_program, startup_program):
            vocab_size, hidden_size = 1000, 512
            seq_len = 128

            # model_a
            model_a = EmbeddingNet(vocab_size, hidden_size)

            x = paddle.static.data(
                name='x', shape=[None, seq_len], dtype='int64'
            )
            y = model_a(x)

            # print(main_program)
            ops = main_program.global_block().ops
            ops = [op.type for op in ops]
            self.assertEqual(ops, ['c_embedding', 'mp_allreduce_sum'])

            weight = model_a.embedding.weight
            self.assertEqual(
                weight.shape,
                (vocab_size // self.model_parallel_size, hidden_size),
            )

    def test_parallel_cross_entropy(self):
        main_program, startup_program = self.get_program()
        with paddle.static.program_guard(main_program, startup_program):
            batch_size = 8
            seq_length = 16
            class_size = 1000
            class_size_per_card = class_size // self.model_parallel_size

            # model_a
            model_a = fleet.meta_parallel.ParallelCrossEntropy()

            x = paddle.static.data(
                name='x', shape=[batch_size, seq_length, class_size_per_card]
            )
            label = paddle.static.data(
                name='label', shape=[batch_size, seq_length], dtype='int64'
            )
            loss_a = model_a(x, label)

            # print(main_program)
            ops = main_program.global_block().ops
            ops = [op.type for op in ops]
            self.assertEqual(
                ops, ['unsqueeze2', 'c_softmax_with_cross_entropy']
            )


if __name__ == '__main__':
    unittest.main()
