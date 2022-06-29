# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

import unittest
import paddle
import numpy as np
import paddle.nn as nn
import paddle.utils as utils
import paddle.static as static
import paddle.nn.functional as F
import paddle.fluid.layers as layers

from paddle.distributed import fleet
from paddle.distributed.auto_parallel.cluster import Cluster
from paddle.distributed.auto_parallel.dist_context import DistributedContext, get_default_distributed_context
from paddle.distributed.auto_parallel.dist_op import DistributedOperator
import paddle.distributed.auto_parallel.dynamic_dims_inference as dyn_dims_infer
from paddle.distributed.auto_parallel.utils import print_program_with_dist_attr

import sys

sys.path.append("..")
import auto_parallel_gpt_model as modeling
from auto_parallel_gpt_model import GPTModel, GPTForPretraining, GPTPretrainingCriterion
from test_cluster import cluster_json

paddle.enable_static()

batch_size = 4
epoch_num = 10
hidden_size = 1024
sequence_len = 512


def get_program():
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    # fleet.init(is_collective=True, strategy=dist_strategy)
    place = paddle.set_device("gpu")
    gpus = [0, 1]
    batch_size = 8
    sequence_len = 512
    vocab_size = 1000

    train_program = static.Program()
    start_program = static.Program()
    modeling.init_global()
    modeling._global_parallel_strategy = "dp_mp_pp"
    modeling.DPMPPP_MESH_LIST = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    with static.program_guard(train_program, start_program):
        tokens = paddle.static.data(name="tokens",
                                    shape=[batch_size, sequence_len],
                                    dtype='int64')
        position_ids = paddle.static.data(name="position_ids",
                                          shape=[batch_size, sequence_len],
                                          dtype='int64')
        attention_mask = paddle.static.data(
            name="attention_mask",
            shape=[batch_size, 1, sequence_len, sequence_len],
            dtype='float32')
        labels = paddle.static.data(name="labels",
                                    shape=[batch_size, sequence_len],
                                    dtype='int64')
        loss_mask = paddle.static.data(name="loss_mask",
                                       shape=[batch_size, sequence_len],
                                       dtype='float32')
        data_holder = [tokens, position_ids, attention_mask, labels, loss_mask]

        gpt = GPTModel(vocab_size=1000,
                       hidden_size=1024,
                       num_hidden_layers=2,
                       num_attention_heads=16,
                       intermediate_size=4 * 1024,
                       hidden_act="gelu",
                       hidden_dropout_prob=0.0,
                       attention_probs_dropout_prob=0.0,
                       max_position_embeddings=1024,
                       type_vocab_size=1,
                       initializer_range=0.02,
                       pad_token_id=0,
                       eos_token_id=7,
                       bos_token_id=0,
                       eol_token_id=3,
                       pp_degree=len(modeling.DPMPPP_MESH_LIST))

        model = GPTForPretraining(gpt,
                                  vocab_size=1000,
                                  hidden_size=64,
                                  initializer_range=0.02)
        preds = model(tokens, position_ids, attention_mask)
        criterion = GPTPretrainingCriterion()
        loss = criterion(preds, labels, loss_mask)

        optimizer = paddle.fluid.optimizer.AdamOptimizer(learning_rate=0.00001,
                                                         beta1=0.9,
                                                         beta2=0.999,
                                                         epsilon=1e-08,
                                                         grad_clip=None)

        feed_vars = {
            "inputs": [tokens, position_ids, attention_mask, loss_mask],
            "labels": [labels]
        }
        fetch_vars = {"loss": [loss]}

    return train_program, start_program, None, loss, optimizer, feed_vars, fetch_vars


class TestDynamicDimensionsInferenceRules(unittest.TestCase):

    def test_fill_constant_batch_size_like(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input = static.data(name="input", shape=[2, 3], dtype='float32')
            output = layers.fill_constant_batch_size_like(input=input,
                                                          value=0,
                                                          shape=[4, 0],
                                                          dtype='float32')
        op = train_program.current_block().ops[0]
        dist_ctx = DistributedContext(train_program, start_program)
        dist_ctx.initialize()
        dist_op = dist_ctx.get_dist_op_for_program(op)
        op_dist_attr = dist_op.dist_attr
        dyn_dims_infer.dynamic_dims_fill_constant_batch_size_like(dist_op)
        self.assertEqual(op_dist_attr.get_output_dynamic_dims(output.name),
                         [0, 1])

    def test_gather(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input = static.data(name="input", shape=[4, 3], dtype='float32')
            index = static.data(name='index', shape=[1], dtype='int32')
            output = layers.gather(input=input, index=index)
        op = train_program.current_block().ops[0]
        dist_ctx = DistributedContext(train_program, start_program)
        dist_ctx.initialize()
        dist_op = dist_ctx.get_dist_op_for_program(op)
        op_dist_attr = dist_op.dist_attr
        op_dist_attr.set_input_dynamic_dims(input.name, [0, 1])
        op_dist_attr.set_output_dynamic_dims(output.name, [1, 0])
        dyn_dims_infer.dynamic_dims_gather(dist_op)
        self.assertEqual(op_dist_attr.get_input_dynamic_dims(input.name),
                         [1, 1])
        self.assertEqual(op_dist_attr.get_output_dynamic_dims(output.name),
                         [1, 1])

    def test_concat(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input0 = static.data(name="input0", shape=[1, 3], dtype='float32')
            input1 = static.data(name="input1", shape=[2, 3], dtype='float32')
            output = layers.concat(input=[input0, input1], axis=0)
        op = train_program.current_block().ops[0]
        dist_ctx = DistributedContext(train_program, start_program)
        dist_ctx.initialize()
        dist_op = dist_ctx.get_dist_op_for_program(op)
        op_dist_attr = dist_op.dist_attr
        op_dist_attr.set_input_dynamic_dims(input0.name, [0, 1])
        op_dist_attr.set_input_dynamic_dims(input1.name, [0, 0])
        dyn_dims_infer.dynamic_dims_concat(dist_op)
        self.assertEqual(op_dist_attr.get_input_dynamic_dims(input1.name),
                         [0, 1])
        self.assertEqual(op_dist_attr.get_output_dynamic_dims(output.name),
                         [0, 1])

    def test_assign(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input = static.data(name="input", shape=[2, 3], dtype='float32')
            output = layers.assign(input)
        op = train_program.current_block().ops[0]
        dist_ctx = DistributedContext(train_program, start_program)
        dist_ctx.initialize()
        dist_op = dist_ctx.get_dist_op_for_program(op)
        op_dist_attr = dist_op.dist_attr
        op_dist_attr.set_output_dynamic_dims(output.name, [1, 1])
        dyn_dims_infer.dynamic_dims_assign(dist_op)
        self.assertEqual(op_dist_attr.get_input_dynamic_dims(input.name),
                         [1, 1])

    def test_scale(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input = static.data(name="input", shape=[2, 3], dtype='float32')
            output = layers.scale(input)
        op = train_program.current_block().ops[0]
        dist_ctx = DistributedContext(train_program, start_program)
        dist_ctx.initialize()
        dist_op = dist_ctx.get_dist_op_for_program(op)
        op_dist_attr = dist_op.dist_attr
        op_dist_attr.set_input_dynamic_dims(input.name, [0, 1])
        dyn_dims_infer.dynamic_dims_scale(dist_op)
        self.assertEqual(op_dist_attr.get_output_dynamic_dims(output.name),
                         [0, 1])

    def test_unsqueeze(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input = static.data(name="input", shape=[2, 3], dtype='float32')
            output = layers.unsqueeze(input, axes=[1])
        op = train_program.current_block().ops[0]
        dist_ctx = DistributedContext(train_program, start_program)
        dist_ctx.initialize()
        dist_op = dist_ctx.get_dist_op_for_program(op)
        op_dist_attr = dist_op.dist_attr
        op_dist_attr.set_input_dynamic_dims(input.name, [0, 1])
        dyn_dims_infer.dynamic_dims_unsqueeze2(dist_op)
        self.assertEqual(op_dist_attr.get_output_dynamic_dims(output.name),
                         [0, 0, 1])

    def test_matmul_v2(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input0 = static.data(name="input0",
                                 shape=[5, 5, 2, 3],
                                 dtype='float32')
            input1 = static.data(name="input1",
                                 shape=[5, 5, 3, 2],
                                 dtype='float32')
            output = paddle.matmul(input0, input1)
        op = train_program.current_block().ops[0]
        dist_ctx = DistributedContext(train_program, start_program)
        dist_ctx.initialize()
        dist_op = dist_ctx.get_dist_op_for_program(op)
        op_dist_attr = dist_op.dist_attr
        op_dist_attr.set_input_dynamic_dims(input1.name, [0, 0, 1, 0])
        dyn_dims_infer.dynamic_dims_matmul_v2(dist_op)
        self.assertEqual(op_dist_attr.get_input_dynamic_dims(input1.name),
                         [0, 0, 0, 1])
        self.assertEqual(op_dist_attr.get_output_dynamic_dims(output.name),
                         [0, 0, 0, 0])

    def test_elementwise_add(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input0 = static.data(name="input0", shape=[2, 3], dtype='float32')
            input1 = static.data(name="input1", shape=[3], dtype='float32')
            output = layers.elementwise_add(input0, input1)
        op = train_program.current_block().ops[0]
        dist_ctx = DistributedContext(train_program, start_program)
        dist_ctx.initialize()
        dist_op = dist_ctx.get_dist_op_for_program(op)
        op_dist_attr = dist_op.dist_attr
        op_dist_attr.set_input_dynamic_dims(input0.name, [1, 0])
        dyn_dims_infer.dynamic_dims_elementwise_add(dist_op)
        self.assertEqual(op_dist_attr.get_output_dynamic_dims(output.name),
                         [1, 0])

        op_dist_attr.set_input_dynamic_dims(input0.name, [0, 0])
        op_dist_attr.set_input_dynamic_dims(input1.name, [1])
        op_dist_attr.set_output_dynamic_dims(output.name, [0, 0])
        dyn_dims_infer.dynamic_dims_elementwise_add(dist_op)
        self.assertEqual(op_dist_attr.get_output_dynamic_dims(output.name),
                         [0, 1])

    def test_softmax(self):
        train_program = static.Program()
        start_program = static.Program()
        with static.program_guard(train_program, start_program):
            input = static.data(name="input", shape=[2, 3], dtype='float32')
            output = layers.softmax(input)
        op = train_program.current_block().ops[0]
        dist_ctx = DistributedContext(train_program, start_program)
        dist_ctx.initialize()
        dist_op = dist_ctx.get_dist_op_for_program(op)
        op_dist_attr = dist_op.dist_attr
        op_dist_attr.set_input_dynamic_dims(input.name, [0, 1])
        dyn_dims_infer.dynamic_dims_softmax(dist_op)
        self.assertEqual(op_dist_attr.get_output_dynamic_dims(output.name),
                         [0, 1])


class TestDynamicDimensionsInference(unittest.TestCase):

    def test_dynamic_dims_inference(self):
        train_program, start_program, dataloader, loss, optimizer, feed_vars, fetch_vars = get_program(
        )
        cluster = Cluster()
        cluster.gen_default_config_cluster(node_count=1, device_count=8)
        dist_context = DistributedContext(train_program, start_program,
                                          optimizer, loss, feed_vars,
                                          fetch_vars, cluster)
        dist_context.initialize()
        dynamic_dims_inference = dyn_dims_infer.DynamicDimensionsInference(
            dist_context)
        dynamic_dims_inference.infer_dynamic_dims()
        print_program_with_dist_attr(train_program, dist_context)


if __name__ == "__main__":
    unittest.main()
