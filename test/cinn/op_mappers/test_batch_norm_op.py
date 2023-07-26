#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

from op_mapper_test import OpMapperTest

import paddle


class TestBatchNormOp(OpMapperTest):
    def init_input_data(self):
        self.num_channels = 16
        self.feed_data = {
            "x": self.random([2, self.num_channels, 8, 8], "float32"),
            'scale': self.random([self.num_channels], "float32"),
            'bias': self.random([self.num_channels], "float32"),
            'mean': self.random([self.num_channels], "float32"),
            'variance': self.random([self.num_channels], "float32"),
        }

    def set_op_type(self):
        return "batch_norm"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype,
        )
        scale = paddle.static.data(
            name='scale',
            shape=self.feed_data['scale'].shape,
            dtype=self.feed_data['scale'].dtype,
        )
        bias = paddle.static.data(
            name='bias',
            shape=self.feed_data['bias'].shape,
            dtype=self.feed_data['bias'].dtype,
        )
        mean = paddle.static.data(
            name='mean',
            shape=self.feed_data['mean'].shape,
            dtype=self.feed_data['mean'].dtype,
        )
        variance = paddle.static.data(
            name='variance',
            shape=self.feed_data['variance'].shape,
            dtype=self.feed_data['variance'].dtype,
        )
        return {
            'X': [x],
            'Scale': [scale],
            'Bias': [bias],
            'Mean': [mean],
            'Variance': [variance],
        }

    def set_op_attrs(self):
        return {
            'epsilon': 1e-5,
            'momentum': 0.9,
            'data_layout': 'NCHW',
            'is_test': False,
            'trainable_statistics': False,
            'use_global_stats': False,
        }

    def set_op_outputs(self):
        return {
            'Y': [self.feed_data['x'].dtype],
            'SavedMean': [self.feed_data['mean'].dtype],
            'SavedVariance': [self.feed_data['variance'].dtype],
        }

    def set_inplace_outputs(self):
        return {'MeanOut': 'Mean', 'VarianceOut': 'Variance'}

    def skip_check_outputs(self):
        # TODO(thisjiang): remove after Variance compute correct
        return {'SavedVariance', 'VarianceOut'}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestBatchNormInferOp(TestBatchNormOp):
    def set_op_attrs(self):
        return {
            'epsilon': 1e-5,
            'momentum': 0.9,
            'data_layout': 'NCHW',
            'is_test': True,
            'trainable_statistics': False,
            'use_global_stats': False,
        }

    def skip_check_outputs(self):
        # 'SavedMean', 'SavedVariance' are None in Paddle
        # TODO(thisjiang): remove after VarianceOut compute correct
        return {'SavedMean', 'SavedVariance', 'VarianceOut'}


if __name__ == "__main__":
    unittest.main()
