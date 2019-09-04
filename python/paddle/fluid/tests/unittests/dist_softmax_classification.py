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
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.initializer import NumpyArrayInitializer
from test_dist_classification_base import DistClassificationRunner
from test_dist_collective_base import runtime_main


# TODO bias attr
class DistSoftmaxClassificationRunner(DistClassificationRunner):
    def __init__(self, args):
        super(DistSoftmaxClassificationRunner, self).__init__(args)
        np.random.seed(1024)
        self.param_value = np.random.rand(args.feature_size, args.class_num)

    def local_classify_subnet(self, feature, label):
        args = self.args
        logits = layers.fc(feature,
                           args.class_num,
                           param_attr=NumpyArrayInitializer(self.param_value))
        loss = layers.softmax_with_cross_entropy(logits, label)
        cost = layers.mean(loss)
        return cost

    def parall_classify_subnet(self, feature, label):
        args = self.args
        shard_dim = (args.class_num + args.nranks - 1) // args.nranks
        shard_start = shard_dim * args.rank
        rank_param_value = self.param_value[:, shard_start:(shard_start +
                                                            shard_dim)]
        cost = layers.collective._distributed_fc_classify(
            x=feature,
            label=label,
            class_num=args.class_num,
            nranks=args.nranks,
            rank_id=args.rank,
            param_attr=NumpyArrayInitializer(rank_param_value))
        return cost


if __name__ == "__main__":
    runtime_main(DistSoftmaxClassificationRunner)
