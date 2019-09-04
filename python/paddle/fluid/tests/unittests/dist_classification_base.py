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
from datetime import datetime

import unittest
import os
import sys
import subprocess
import six
import argparse
import pickle
import numpy as np
import paddle.fluid as fluid

from paddle.fluid.transpiler.collective import \
    GradAllReduce, DistributedClassificationOptimizer
from test_dist_collective_base import DistCollectiveRunner, elog

DEFAULT_FEATURE_SIZE = 4
DEFAULT_CLASS_NUM = 4


class DistClassificationRunner(DistCollectiveRunner):
    ##################################
    ##### user specified methods #####

    @classmethod
    def add_other_arguments(cls, parser):
        pass

    def local_classify_subnet(self, feature, label):
        raise NotImplementedError(
            'local_classifiy_subnet should be implemented by child classes.')

    def parall_classify_subnet(self, feature, label):
        raise NotImplementedError(
            'parall_classify_subnet should be implemented by child classes.')

    ##### user specified methods #####
    ##################################

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            '--feature_size', type=int, default=DEFAULT_FEATURE_SIZE)
        parser.add_argument('--class_num', type=int, default=DEFAULT_CLASS_NUM)
        cls.add_other_arguments(parser)

    def build_local_net(self):
        return self.build_classification_net()

    def build_parall_net(self):
        return self.build_classification_net()

    def yield_sample(self, np_random):
        yield [
            np_random.rand(self.args.feature_size),
            np_random.randint(self.args.class_num)
        ]

    def dist_optimize(self, optimizer, loss):
        args = self.args
        optimizer_wrapper = DistributedClassificationOptimizer(optimizer,
                                                               args.batch_size)
        optimizer_wrapper.minimize(loss)
        transpiler = GradAllReduce()
        transpiler.transpile(
            rank=args.rank,
            endpoints=args.endpoints,
            current_endpoint=args.current_endpoint,
            wait_port=True)

    def build_classification_net(self):
        args = self.args
        feature = fluid.layers.data(
            name='feature', shape=[args.feature_size], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        if args.nranks <= 1:
            elog(self, 'build local network')
            loss = self.local_classify_subnet(feature, label)
        else:
            elog(self, 'build parallel network')
            loss = self.parall_classify_subnet(feature, label)
        return [feature, label], loss
