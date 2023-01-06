# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.fluid as fluid

paddle.enable_static()


class TestFleetFP16CompressOptimizer(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"

    def net(self, main_prog, startup_prog, dtype='float32'):
        with fluid.program_guard(main_prog, startup_prog):
            input_x = paddle.fluid.layers.data(
                name="x", shape=[32], dtype=dtype
            )
            input_y = paddle.fluid.layers.data(
                name="y", shape=[1], dtype='int64'
            )

            fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
            fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
            prediction = paddle.fluid.layers.fc(
                input=[fc_2], size=2, act='softmax'
            )
            cost = paddle.nn.functional.cross_entropy(
                input=prediction,
                label=input_y,
                reduction='none',
                use_softmax=False,
            )
            avg_cost = paddle.mean(x=cost)

            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.fp16_allreduce = True
        return avg_cost, strategy

    def test_fp16_allreduce_optimizer(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)

        optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        ops = [op.type for op in avg_cost.block.ops]
        cast_out = [
            op.output('Out')[0]
            for op in avg_cost.block.ops
            if op.type == 'cast'
        ]

        cast_op_count = 0
        for name in ops:
            if name == 'cast':
                cast_op_count += 1
        self.assertIn('cast', ops)
        self.assertEqual(cast_op_count, 12)  # 6 + 6, cast_fp16 + cast_fp32

        for name in cast_out:
            self.assertIn('cast_fp16', name)

    def test_fp16_allreduce_not_apply_fp16_net(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog, dtype='float16')

        optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertNotIn('cast', ops)


if __name__ == "__main__":
    unittest.main()
