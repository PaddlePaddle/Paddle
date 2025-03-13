# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from inference_pass_test import InferencePassTest

import paddle
from paddle.framework import core
from paddle.inference import Config, create_predictor


class TestNet(paddle.nn.Layer):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.sp_conv = paddle.sparse.nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=True,
        )
        self.sp_bn = paddle.sparse.nn.BatchNorm(
            out_planes, epsilon=1e-3, momentum=1 - 0.01, data_format='NHWC'
        )

    def forward(self, indices, values):
        x = paddle.sparse.sparse_coo_tensor(
            indices=indices,
            values=values,
            shape=[1, 32, 32, 3],
            dtype='float32',
        )
        x = self.sp_conv(x)
        x = self.sp_bn(x)
        return x.to_dense()


class SparseConvUsingBuffer(InferencePassTest):
    def setUp(self):
        paddle.disable_static()
        self.test_model = TestNet(3, 3)
        self.test_values = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).astype(
            'float32'
        )
        self.test_indices = np.array(
            [[0, 0, 0], [0, 16, 16], [0, 20, 8]]
        ).astype('int32')
        self.out_baseline = self.test_model(
            paddle.to_tensor(self.test_indices, stop_gradient=False),
            paddle.to_tensor(self.test_values, stop_gradient=False),
        ).flatten()

        self.path_prefix = "inference_test_models/sparse_conv_using_buffer"
        self.cache_dir = "inference_test_models/cache"
        paddle.jit.save(
            self.test_model,
            self.path_prefix,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[3, -1], dtype='int32', name="indices"
                ),
                paddle.static.InputSpec(
                    shape=[-1, 3], dtype='float32', name="values"
                ),
            ],
        )

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            out_check = self.inference()
            np.testing.assert_allclose(
                self.out_baseline, out_check, rtol=1e-5, atol=1e-2
            )

    def inference(self):
        # Config
        config = Config(
            self.path_prefix + ".json", self.path_prefix + ".pdiparams"
        )
        config.enable_use_gpu(100, 0)
        config.set_optim_cache_dir(self.cache_dir)
        config.exp_sparse_conv_using_buffer([[3, 3]], [[2, 2]])

        # predictor
        predictor = create_predictor(config)

        # inference
        values_tensor = predictor.get_input_handle("values")
        indices_tensor = predictor.get_input_handle("indices")

        values_tensor.reshape(self.test_values.shape)
        indices_tensor.reshape(self.test_indices.shape)

        values_tensor.copy_from_cpu(self.test_values.copy())
        indices_tensor.copy_from_cpu(self.test_indices.copy())

        predictor.run()
        output_tensor = predictor.get_output_handle(
            predictor.get_output_names()[0]
        )
        out = output_tensor.copy_to_cpu()
        out = np.array(out).flatten()
        return out


if __name__ == "__main__":
    unittest.main()
