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
import unittest

from test_cluster import cluster_json

from paddle.distributed.auto_parallel.static.cluster import Cluster
from paddle.distributed.auto_parallel.static.cost.comp_op_cost import (
    AssignOpCost,
    AssignValueOpCost,
    BeamSearchDecodeOpCost,
    BeamSearchOpCost,
    CastOpCost,
    ConcatOpCost,
    DropoutGradOpCost,
    ElementwiseAddGradOpCost,
    ElementwiseAddOpCost,
    ElementwiseDivGradOpCost,
    ElementwiseDivOpCost,
    ElementwiseMulGradOpCost,
    ElementwiseMulOpCost,
    ElementwiseSubOpCost,
    EmbeddingGradOpCost,
    EmbeddingOpCost,
    FillConstantBatchSizeLikeOpCost,
    FillConstantOpCost,
    FusedSoftmaxMaskUpperTriangleGradOpCost,
    FusedSoftmaxMaskUpperTriangleOpCost,
    GatherOpCost,
    GeluGradOpCost,
    GeluOpCost,
    GreaterEqualOpCost,
    IncrementOpCost,
    IsEmptyOpCost,
    LayerNormGradOpCost,
    LayerNormOpCost,
    LessThanOpCost,
    LodResetOpCost,
    LogicalAndOpCost,
    LogicalNotOpCost,
    LogOpCost,
    LookupTableV2GradOpCost,
    LookupTableV2OpCost,
    MatmulOpCost,
    MatmulV2GradOpCost,
    MatmulV2OpCost,
    MemcpyOpCost,
    MulGradOpCost,
    MulOpCost,
    OneHotOpCost,
    ReadFromArrayOpCost,
    ReduceMeanGradOpCost,
    ReduceMeanOpCost,
    ReduceSumGradOpCost,
    ReduceSumOpCost,
    Reshape2GradOpCost,
    Reshape2OpCost,
    ScaleOpCost,
    SliceOpCost,
    SoftmaxGradOpCost,
    SoftmaxOpCost,
    SoftmaxWithCrossEntropyGradOpCost,
    SoftmaxWithCrossEntropyOpCost,
    SplitOpCost,
    SquareGradOpCost,
    SquareOpCost,
    Squeeze2OpCost,
    SumOpCost,
    TopKOpCost,
    Transpose2GradOpCost,
    Transpose2OpCost,
    Unsqueeze2OpCost,
    WriteToArrayOpCost,
)


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

        op_cost = MatmulV2GradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = MemcpyOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = MulOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = MulGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = OneHotOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ReadFromArrayOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ReduceSumOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ReduceSumGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = Reshape2OpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = MatmulV2OpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = Reshape2GradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ReduceMeanOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ReduceMeanGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = ScaleOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = SliceOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = SoftmaxOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = SoftmaxGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = SoftmaxWithCrossEntropyOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = SoftmaxWithCrossEntropyGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = SplitOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = Squeeze2OpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = SquareOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = SquareGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = SumOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = TopKOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = Transpose2OpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = Transpose2GradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = Unsqueeze2OpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = WriteToArrayOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = DropoutGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = FusedSoftmaxMaskUpperTriangleOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        op_cost = FusedSoftmaxMaskUpperTriangleGradOpCost(cluster=cluster)
        self.assertTrue(op_cost.flops >= 0)
        self.assertTrue(op_cost.time >= 0)
        self.assertTrue(op_cost.memory >= 0)

        # Remove unnecessary files
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)


if __name__ == "__main__":
    unittest.main()
