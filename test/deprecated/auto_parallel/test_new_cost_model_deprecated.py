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

import json
import os
import sys
import tempfile
import unittest

sys.path.append("../../auto_parallel")
from test_cluster import cluster_json

import paddle
import paddle.distributed.auto_parallel.static.cost as cost_model
from paddle.distributed.auto_parallel.static.cluster import Cluster
from paddle.distributed.auto_parallel.static.cost import CommContext
from paddle.distributed.auto_parallel.static.cost.base_cost import (
    build_comp_desc_from_op,
    build_comp_desc_str_for_predict,
    calc_time_by_modeling,
)

paddle.enable_static()


def check_cost(cost):
    if cost.memory >= 0 and cost.flops >= 0 and cost.time >= 0:
        return True
    return False


class TestCost(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

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
        matmul_v2_cost = cost_model._g_op_cost_factory["matmul_v2"](
            op=matmul_v2_op
        )
        desc = build_comp_desc_from_op(op=matmul_v2_op)
        desc_str = build_comp_desc_str_for_predict(desc)
        self.assertIsNotNone(desc_str)
        self.assertTrue(check_cost(matmul_v2_cost.cost))
        time = calc_time_by_modeling(op=matmul_v2_op)
        self.assertEqual(time, matmul_v2_cost.cost.time)
        tensor_cost = cost_model.TensorCost(tensor=x)
        # check memory
        self.assertEqual(tensor_cost.cost.memory, 1600)

    def test_comm_cost(self):
        # Build cluster
        cluster_json_path = os.path.join(
            self.temp_dir.name, "auto_parallel_cluster.json"
        )
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)

        # Build CommContext
        CommContext._has_instance = None
        CommContext._instance = None
        comm_context = CommContext(cluster)
        desc = {}
        desc["op"] = "c_allreduce_sum"
        desc["inputs"] = {"X": [(paddle.float32, [100, 200])]}
        desc["group_ranks"] = [0, 1]
        allreduce_cost = cost_model._g_op_cost_factory["c_allreduce_sum"](
            op_desc=desc, comm_context=CommContext(cluster)
        )
        self.assertTrue(check_cost(allreduce_cost.cost))

        # Remove unnecessary files
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)

    def test_cost_estimator(self):
        # Build cluster
        cluster_json_path = os.path.join(
            self.temp_dir.name, "auto_parallel_cluster.json"
        )
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)

        train_program = paddle.static.Program()
        cost_estimator = cost_model.CostEstimator(
            train_program, cluster=cluster
        )
        self.assertIsNotNone(cost_estimator)

        # Remove unnecessary files
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)


if __name__ == "__main__":
    unittest.main()
