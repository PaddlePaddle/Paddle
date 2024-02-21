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

import math
import os
import unittest

import numpy as np

import paddle
from paddle.base.layer_helper import LayerHelper

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.random.seed(123)
paddle.seed(123)


def xft_weight_quantize(weight, alog, bits=paddle.int8):
    helper = LayerHelper('xft_weight_quantize', **locals())
    inputs = {
        'x': [weight],
    }
    attrs = {
        'algo': alog,
    }
    out = helper.create_variable_for_type_inference(bits)
    scale = helper.create_variable_for_type_inference(paddle.float32)
    zero_point = helper.create_variable_for_type_inference(paddle.float32)
    helper.append_op(
        type='xft_weight_quantize',
        inputs=inputs,
        outputs={'out': out, 'scale': scale, 'zero_point': zero_point},
        attrs=attrs,
    )
    return out, scale, zero_point


def xft_weight_only_linear(
    x, weight_quant, scale, zero_point, bias=None, bits="int8"
):
    helper = LayerHelper("xft_weight_only_linear", **locals())
    dtype = x.dtype
    inputs = {
        "x": [x],
        "weight": [weight_quant],
        "weight_scale": [scale],
        "weight_zero_point": [zero_point],
    }
    if bias is not None:
        inputs["bias"] = [bias]
    attrs = {"weight_dtype": bits}

    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="xft_weight_only_linear",
        inputs=inputs,
        outputs={"out": out},
        attrs=attrs,
    )
    return out


class XFTWeightOnlyLinearTestCase(unittest.TestCase):
    def test_weight_only_linear(self):
        paddle.set_device("cpu")
        x = paddle.rand(shape=(16, 1, 4096), dtype='float32').cpu()
        x = x * (1 / math.sqrt(4096))
        weight = paddle.rand(shape=(4096, 12288), dtype='float32').cpu()
        weight = weight * (1 / math.sqrt(4096))

        weight_quant, scale, zero_point = xft_weight_quantize(
            weight, "weight_only_int8", paddle.int8
        )
        act_out = xft_weight_only_linear(
            x, weight_quant, scale, zero_point, None
        )
        base_out = paddle.matmul(x=x, y=weight)
        np.testing.assert_allclose(act_out, base_out, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
