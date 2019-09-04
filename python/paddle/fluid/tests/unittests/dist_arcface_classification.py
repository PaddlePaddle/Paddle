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
import paddle.fluid.layers.collective as collective
from paddle.fluid.initializer import NumpyArrayInitializer
from test_dist_classification_base import DistClassificationRunner, runtime_main


# TODO donot transpose weight
class DistArcfaceClassificationRunner(DistClassificationRunner):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--arcface_margin', type=float, default=0.0)
        parser.add_argument('--arcface_scale', type=float, default=1.0)

    def __init__(self, args):
        super(DistArcfaceClassificationRunner, self).__init__(args)
        np.random.seed(1024)
        self.param_value = np.random.rand(args.class_num, args.feature_size)

    def local_classify_subnet(self, feature, label):
        args = self.args

        weight = layers.create_parameter(
            dtype=feature.dtype,
            shape=[args.class_num, args.feature_size],
            default_initializer=NumpyArrayInitializer(self.param_value),
            is_bias=False)

        # normalize feature
        feature_l2 = layers.sqrt(
            layers.reduce_sum(
                layers.square(feature), dim=1))
        norm_feature = layers.elementwise_div(feature, feature_l2, axis=0)

        # normalize weight
        weight_l2 = layers.sqrt(layers.reduce_sum(layers.square(weight), dim=1))
        norm_weight = layers.elementwise_div(weight, weight_l2, axis=0)
        norm_weight = layers.transpose(norm_weight, perm=[1, 0])

        cos = layers.mul(norm_feature, norm_weight)

        theta = layers.acos(cos)
        margin_cos = layers.cos(theta + args.arcface_margin)

        one_hot = layers.one_hot(label, depth=args.class_num)

        diff = (margin_cos - cos) * one_hot
        target_cos = cos + diff
        logit = layers.scale(target_cos, scale=args.arcface_scale)

        loss = layers.softmax_with_cross_entropy(logit, label)
        cost = layers.mean(loss)

        return cost

    def parall_classify_subnet(self, feature, label):
        args = self.args
        shard_dim = (args.class_num + args.nranks - 1) // args.nranks
        shard_start = shard_dim * args.rank
        rank_param_value = self.param_value[shard_start:(shard_start + shard_dim
                                                         ), :]
        cost = layers.collective._distributed_arcface_classify(
            x=feature,
            label=label,
            class_num=args.class_num,
            nranks=args.nranks,
            rank_id=args.rank,
            margin=args.arcface_margin,
            logit_scale=args.arcface_scale,
            param_attr=NumpyArrayInitializer(rank_param_value))
        return cost


if __name__ == "__main__":
    runtime_main(DistArcfaceClassificationRunner)
