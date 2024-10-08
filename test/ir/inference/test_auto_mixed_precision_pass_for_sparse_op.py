# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.inference import Config, PrecisionType, create_predictor


class TestNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.sp_conv = paddle.sparse.nn.SubmConv2D(
            3,
            3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
            key=None,
        )
        self.sp_bn = paddle.sparse.nn.BatchNorm(
            3, epsilon=1e-3, momentum=1 - 0.01, data_format='NHWC'
        )
        self.relu = paddle.sparse.nn.ReLU()

    def forward(self, indices, values):
        x = paddle.sparse.sparse_coo_tensor(
            indices=indices,
            values=values,
            shape=[1, 32, 32, 3],
            dtype='float32',
        )
        x = self.sp_conv(x)
        x = self.sp_bn(x)
        x = self.relu(x)
        return x.to_dense()


class AutoMixedPrecisionPassForSparseOp(InferencePassTest):
    def setUp(self):
        paddle.disable_static()
        self.test_model = TestNet()
        self.values = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).astype(
            'float32'
        )
        self.indices = np.array([[0, 0, 0], [0, 16, 16], [0, 20, 8]]).astype(
            "int32"
        )
        with paddle.pir_utils.OldIrGuard():
            self.path_prefix = "inference_test_models/auto_mixed_precision_pass_for_sparse_op_test"
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
        fp32_out = self.inference("fp32")
        fp16_out = self.inference("fp16")
        np.testing.assert_allclose(fp32_out, fp16_out, rtol=1e-5, atol=1e-2)

    def inference(self, precision):
        # Config
        config = Config(
            self.path_prefix + ".pdmodel", self.path_prefix + ".pdiparams"
        )
        if precision == "fp16":
            config.enable_use_gpu(100, 0, PrecisionType.Half)
            white_list = ["sparse_batch_norm", "sparse_relu"]
            config.exp_enable_mixed_precision_ops(set(white_list))
        else:
            config.enable_use_gpu(100, 0, PrecisionType.Float32)

        # predictor
        predictor = create_predictor(config)

        # inference
        indices_tensor = predictor.get_input_handle("indices")
        indices_tensor.reshape(self.indices.shape)
        indices_tensor.copy_from_cpu(self.indices.copy())
        values_tensor = predictor.get_input_handle("values")
        values_tensor.reshape(self.values.shape)
        values_tensor.copy_from_cpu(self.values.copy())
        predictor.run()
        output_tensor = predictor.get_output_handle(
            predictor.get_output_names()[0]
        )
        out = output_tensor.copy_to_cpu()
        out = np.array(out).flatten()
        return out


if __name__ == "__main__":
    unittest.main()
