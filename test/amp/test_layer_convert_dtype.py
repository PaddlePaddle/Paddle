# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.base import core


class MyModel(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = paddle.nn.Linear(input_size, hidden_size)
        self.linear2 = paddle.nn.Linear(hidden_size, hidden_size)
        self.linear3 = paddle.nn.Linear(hidden_size, 1)
        self.batchnorm = paddle.nn.Sequential(paddle.nn.BatchNorm(hidden_size))
        register_buffer_in_temp = paddle.ones([4, 6])
        self.register_buffer('register_buffer_in', register_buffer_in_temp)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.batchnorm(x)
        x = self.linear3(x)
        return x


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
    "Require compiled with CUDA or XPU.",
)
class TestDtypeConvert(unittest.TestCase):
    def setUp(self):
        self.batch_size, self.input_size, self.hidden_size = 128, 128, 256

    def verify_trans_dtype(
        self, test_type=None, excluded_layers=None, corrected_dtype=None
    ):
        model = MyModel(self.input_size, self.hidden_size)
        if test_type == 'float16':
            model.float16(excluded_layers=excluded_layers)
        elif test_type == 'bfloat16':
            model.bfloat16(excluded_layers=excluded_layers)
        else:
            model.float(excluded_layers=excluded_layers)

        for name, para in model.named_parameters():
            if 'linear' in name:
                self.assertEqual(para.dtype, corrected_dtype)
            elif 'batchnorm' in name:
                if excluded_layers is None:
                    self.assertEqual(para.dtype, paddle.float32)
                else:
                    self.assertEqual(para.dtype, paddle.float16)

    def test_excluded_layers(self):
        self.verify_trans_dtype(
            test_type='float16',
            excluded_layers=[nn.Linear],
            corrected_dtype=paddle.float32,
        )
        self.verify_trans_dtype(
            test_type='float16',
            excluded_layers=nn.Linear,
            corrected_dtype=paddle.float32,
        )

    def test_float16(self):
        self.verify_trans_dtype(
            test_type='float16',
            corrected_dtype=paddle.float16,
        )

    @unittest.skipIf(
        not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
        "Require compiled with CUDA or XPU.",
    )
    @unittest.skipIf(
        core.is_compiled_with_cuda()
        and paddle.device.cuda.get_device_capability()[0] >= 8.0,
        "run test when maximum gpu's compute capability is 8.0.",
    )
    @unittest.skipIf(
        core.is_compiled_with_xpu()
        and core.get_xpu_device_version(0) >= core.XPUVersion.XPU3,
        "run test when xpu's compute capability < xpu3.",
    )
    def test_unsupported_bfloat16(self):
        self.verify_trans_dtype(
            test_type='bfloat16',
            corrected_dtype=paddle.float32,
        )

    @unittest.skipIf(
        not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
        "Require compiled with CUDA or XPU.",
    )
    @unittest.skipIf(
        core.is_compiled_with_cuda()
        and paddle.device.cuda.get_device_capability()[0] < 8.0,
        "run test when gpu's compute capability is at least 8.0.",
    )
    @unittest.skipIf(
        core.is_compiled_with_xpu()
        and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
        "run test when xpu's compute capability >= xpu3.",
    )
    def test_supported_bfloat16(self):
        self.verify_trans_dtype(
            test_type='bfloat16',
            corrected_dtype=paddle.bfloat16,
        )

    def test_float32(self):
        paddle.set_default_dtype('float16')
        self.verify_trans_dtype(
            test_type='float32',
            corrected_dtype=paddle.float32,
        )
        paddle.set_default_dtype('float32')

    def test_excluded_layers_type_error(self):
        self.assertRaises(
            TypeError, self.verify_trans_dtype, excluded_layers=111
        )


class TestSupportedTypeInfo(unittest.TestCase):
    def test_cpu(self):
        res = paddle.amp.is_float16_supported('cpu')
        self.assertEqual(res, False)
        res = paddle.amp.is_bfloat16_supported('cpu')
        self.assertEqual(res, core.supports_bfloat16())

    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "Require compiled with CUDA."
    )
    def test_gpu_fp16_supported(self):
        res = paddle.amp.is_float16_supported()
        self.assertEqual(res, True)
        res = paddle.amp.is_float16_supported('gpu')
        self.assertEqual(res, True)
        res = paddle.amp.is_float16_supported('gpu:0')
        self.assertEqual(res, True)

    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability()[0] >= 8.0,
        "run test when maximum gpu's compute capability is 8.0.",
    )
    def test_gpu_bf16_unsupported(self):
        res = paddle.amp.is_bfloat16_supported()
        self.assertEqual(res, False)
        res = paddle.amp.is_bfloat16_supported('gpu')
        self.assertEqual(res, False)

    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability()[0] < 8.0,
        "run test when gpu's compute capability is at least 8.0.",
    )
    def test_gpu_bf16_supported(self):
        res = paddle.amp.is_bfloat16_supported()
        self.assertEqual(res, True)
        res = paddle.amp.is_bfloat16_supported('gpu')
        self.assertEqual(res, True)

    def test_device_value_error(self):
        self.assertRaises(
            ValueError, paddle.amp.is_float16_supported, device='xxx'
        )
        self.assertRaises(
            ValueError, paddle.amp.is_float16_supported, device=111
        )

    @unittest.skipIf(
        not core.is_compiled_with_xpu()
        or not core.get_xpu_device_version(0) >= core.XPUVersion.XPU2,
        "run test when xpu's compute capability >= xpu2.",
    )
    def test_xpu_fp16_supported(self):
        res = paddle.amp.is_float16_supported()
        self.assertEqual(res, True)
        res = paddle.amp.is_float16_supported('xpu')
        self.assertEqual(res, True)
        res = paddle.amp.is_float16_supported('xpu:0')
        self.assertEqual(res, True)

    @unittest.skipIf(
        not core.is_compiled_with_xpu()
        or core.get_xpu_device_version(0) >= core.XPUVersion.XPU3,
        "run test when xpu's compute capability < xpu3.",
    )
    def test_xpu_bf16_unsupported(self):
        res = paddle.amp.is_bfloat16_supported()
        self.assertEqual(res, False)
        res = paddle.amp.is_bfloat16_supported('xpu')
        self.assertEqual(res, False)

    @unittest.skipIf(
        not core.is_compiled_with_xpu()
        or not core.get_xpu_device_version(0) >= core.XPUVersion.XPU3,
        "run test when xpu's compute capability >= xpu3.",
    )
    def test_xpu_bf16_supported(self):
        res = paddle.amp.is_bfloat16_supported()
        self.assertEqual(res, True)
        res = paddle.amp.is_bfloat16_supported('xpu')
        self.assertEqual(res, True)


if __name__ == '__main__':
    unittest.main()
