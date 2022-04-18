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
# limitations under the License.

import unittest

import paddle
import paddle.distributed.auto_parallel.cost as cost_model
from paddle.distributed.auto_parallel.cost.base_cost import parse_to_desc
from paddle.distributed.auto_parallel.cost.base_cost import parse_desc_to_str
from paddle.distributed.auto_parallel.cost.base_cost import calc_time_from_model

paddle.enable_static()


def check_cost(cost):
    if cost.memory >= 0 and cost.flops >= 0 and cost.time >= 0:
        return True
    return False


class TestCost(unittest.TestCase):
    def test_base_cost(self):
        cost = cost_model.Cost(memory=100, flops=200, time=0.5)
        self.assertTrue(check_cost(cost))

    def test_comp_cost(self):
        x = paddle.static.data(name="x", shape=[20, 20], dtype='float32')
        y = paddle.static.data(name="y", shape=[20, 20], dtype='float32')

        z = paddle.matmul(x, y)
        matmul_v2_op = None
        ops = paddle.static.default_main_program().global_block().ops
        for op in ops:
            if op.type == "matmul_v2":
                matmul_v2_op = op
                break
        matmul_v2_cost = cost_model.OP_COST_FACTORY["matmul_v2"](
            op=matmul_v2_op)
        desc = parse_to_desc(op=matmul_v2_op)
        desc_str = parse_desc_to_str(desc)
        self.assertIsNotNone(desc_str)
        self.assertTrue(check_cost(matmul_v2_cost.cost))
        time = calc_time_from_model(op=matmul_v2_op)
        self.assertEqual(time, matmul_v2_cost.cost.time)
        tensor_cost = cost_model.TensorCost(tensor=x)
        # check memory
        self.assertEqual(tensor_cost.cost.memory, 1600)

    def test_comm_cost(self):
        desc = {}
        desc["op"] = "c_allreduce_sum"
        desc["inputs"] = {"X": [([100, 200], paddle.float32)]}
        allreduce_cost = cost_model.OP_COST_FACTORY["c_allreduce_sum"](
            op_desc=desc)
        self.assertTrue(check_cost(allreduce_cost.cost))

    def test_cost_estimator(self):
        train_program = paddle.static.Program()
        cost_estimator = cost_model.CostEstimator(train_program)
        self.assertIsNotNone(cost_estimator)


if __name__ == "__main__":
    unittest.main()
