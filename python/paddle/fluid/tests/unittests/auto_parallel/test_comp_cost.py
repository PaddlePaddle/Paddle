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
import os
import json

import paddle
from paddle.distributed.auto_parallel.cluster import Cluster
from paddle.distributed.auto_parallel.cost.comp_op_cost import AssignOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import AssignValueOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import BeamSearchOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import BeamSearchDecodeOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import CastOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import ConcatOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import ElementwiseAddOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import ElementwiseAddGradOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import ElementwiseDivOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import ElementwiseDivGradOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import ElementwiseMulOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import ElementwiseMulGradOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import ElementwiseSubOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import EmbeddingOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import EmbeddingGradOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import FillConstantOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import FillConstantBatchSizeLikeOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import FillConstantBatchSizeLikeGradOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import GatherOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import GeluOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import GeluGradOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import GreaterEqualOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import IncrementOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import IsEmptyOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import LayerNormOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import LayerNormGradOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import LessThanOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import LogicalNotOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import LogicalAndOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import LodResetOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import LogOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import LookupTableV2OpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import LookupTableV2GradOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import MatmulOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import MatmulGradOpCost
from paddle.distributed.auto_parallel.cost.comp_op_cost import MatmulV2OpCost

from test_cluster import cluster_json


class TestCompOpCost(unittest.TestCase):
    def test_comp_cost(self):
        # Build cluster
        file_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_json_path = os.path.join(file_dir, "auto_parallel_cluster.json")
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, "w") as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)

        op_cost = AssignOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = AssignValueOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = BeamSearchOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = BeamSearchDecodeOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = CastOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ConcatOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ElementwiseAddOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ElementwiseAddGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ElementwiseDivOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ElementwiseDivGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ElementwiseMulOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ElementwiseMulGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ElementwiseSubOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = EmbeddingOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = EmbeddingGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = FillConstantOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = FillConstantBatchSizeLikeOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = FillConstantBatchSizeLikeGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = GatherOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = GeluOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = GeluGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = GreaterEqualOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = IncrementOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = IsEmptyOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = LayerNormOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = LayerNormGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = LessThanOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = LogicalNotOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = LogicalAndOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = LodResetOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = LogOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = LookupTableV2OpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = LookupTableV2GradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = MatmulOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = MatmulV2OpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        # Remove unnecessary files
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)


if __name__ == "__main__":
    unittest.main()
