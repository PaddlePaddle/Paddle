#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import convert_float_to_uint16
from scipy.special import erf, expit

import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.incubate.nn.functional import fused_bias_act


def round_type_1_process(val):
    dtype = type(val)
    if val >= 0:
        return dtype(np.floor(val + 0.5))
    return dtype(np.ceil(val - 0.5))


# rounding to nearest ties away from zero
round_type_1 = np.vectorize(round_type_1_process)

M_SQRT1_2 = 0.70710678118654752440


def gelu(x):
    out = (
        0.5 * x.astype('float32') * (1.0 + erf(x.astype('float32') * M_SQRT1_2))
    )
    return out.astype(x.dtype)


def swish(x):
    out = x.astype('float32') * expit(x.astype('float32'))
    return out.astype(x.dtype)


def fake_dequant(values, dequant_scales):
    out = values * dequant_scales.astype('float32')
    return out


def fake_quant(
    values, shift, smooth, quant_scale, max_bound, min_bound, round_type
):
    values_tmp = (values + shift) * smooth
    values_tmp = max_bound * quant_scale * values_tmp
    if round_type == 0:
        values_tmp = np.rint(values_tmp)
    elif round_type == 1:
        values_tmp = round_type_1(values_tmp)
    return np.clip(values_tmp, min_bound, max_bound).astype(np.int8)


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_rocm(),
    "core is not compiled with CUDA or ROCm",
)
class TestFusedBiasActOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2017)
        np.random.seed(2017)

        self.op_type = "fused_bias_act"
        self.rtol = 1e-5
        self.atol = 1e-3

        self.batch_size = 2
        self.seq_len = 20
        self.cols = 512

        self.dtype = 'float32'
        self.act_method = 'gelu'
        self.compute_dtype = 'default'

        self.use_glu = False

        self.init_test_case()
        self.generate_inputs()

    def init_test_case(self):
        pass

    def generate_inputs(self):
        self.x = (
            np.random.rand(self.batch_size, self.seq_len, self.cols) * 16
        ).astype(self.dtype)
        self.bias = np.random.rand(self.cols).astype(self.dtype)

    def compute_baseline_output(self):
        out = gelu(self.x + self.bias).astype(self.dtype)
        return out

    def compute_paddle_output(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x = paddle.to_tensor(self.x)
        bias = paddle.to_tensor(self.bias)

        return fused_bias_act(
            x=x,
            bias=bias,
            act_method=self.act_method,
            compute_dtype=self.compute_dtype,
        )

    def test_check_output(self):
        final_out_ref = self.compute_baseline_output()
        final_out = self.compute_paddle_output()
        np.testing.assert_allclose(
            final_out_ref, final_out, rtol=self.rtol, atol=self.atol
        )


class TestBaseFP16(TestFusedBiasActOp):
    def init_test_case(self):
        self.dtype = np.float16
        self.act_method = 'gelu'


class TestWithComTypeFP32(TestFusedBiasActOp):
    def init_test_case(self):
        self.dtype = 'float32'
        self.act_method = 'gelu'
        self.compute_dtype = 'fp32'


class TestWithComTypeFP16(TestFusedBiasActOp):
    def init_test_case(self):
        self.dtype = 'float16'
        self.act_method = 'gelu'
        self.compute_dtype = 'fp16'


class TestFastGeluFP16(TestFusedBiasActOp):
    def use_fast_math(self, enabled):
        paddle.set_flags({'FLAGS_use_fast_math': enabled})

    def init_test_case(self):
        self.dtype = np.float16
        self.act_method = 'gelu'

    def compute_baseline_output(self):
        out = F.gelu(
            paddle.to_tensor(self.x) + paddle.to_tensor(self.bias),
            approximate=True,
        )
        return out

    def compute_paddle_output(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x = paddle.to_tensor(self.x)
        bias = paddle.to_tensor(self.bias)
        self.use_fast_math(True)
        out = fused_bias_act(
            x=x,
            bias=bias,
            act_method=self.act_method,
        )
        self.use_fast_math(False)
        return out


class TestGegluFP16(TestFusedBiasActOp):
    def init_test_case(self):
        self.dtype = np.float16
        self.act_method = 'geglu'

    def compute_baseline_output(self):
        res_tmp = (self.x + self.bias).astype(self.dtype)
        res_tmp_head = res_tmp[:, :, : self.cols // 2]
        res_tmp_tail = res_tmp[:, :, self.cols // 2 :]
        res_tmp_head_act = gelu(res_tmp_head)
        out = res_tmp_head_act * res_tmp_tail
        return out


class TestSwigluFP16(TestFusedBiasActOp):
    def init_test_case(self):
        self.dtype = np.float16
        self.act_method = 'swiglu'

    def compute_baseline_output(self):
        res_tmp = (self.x + self.bias).astype(self.dtype)
        res_tmp_head = res_tmp[:, :, : self.cols // 2]
        res_tmp_tail = res_tmp[:, :, self.cols // 2 :]
        res_tmp_head_act = swish(res_tmp_head)
        out = res_tmp_head_act * res_tmp_tail
        return out


class TestQuantFP32(TestFusedBiasActOp):
    def init_test_case(self):
        self.atol = 1

        self.dtype = 'float32'
        self.compute_dtype = 'fp32'
        self.quant_scale = 0.5
        self.quant_round_type = 1
        self.quant_max_bound = 127.0
        self.quant_min_bound = -127.0

    def generate_inputs(self):
        self.x = np.random.randint(
            low=-16, high=16, size=(self.batch_size, self.seq_len, self.cols)
        ).astype('int32')
        self.bias = np.random.rand(self.cols).astype(self.dtype)
        self.dequant_scales = np.random.rand(self.cols).astype('float32')
        quant_params_cols = self.cols // 2 if self.use_glu else self.cols
        self.shift = np.zeros(quant_params_cols).astype(self.dtype)
        self.smooth = np.ones(quant_params_cols).astype(self.dtype)

    def compute_baseline_output(self):
        input_dequanted = fake_dequant(self.x, self.dequant_scales)
        output_tmp = gelu(input_dequanted + self.bias).astype(self.dtype)
        out = fake_quant(
            output_tmp,
            self.shift,
            self.smooth,
            self.quant_scale,
            self.quant_max_bound,
            self.quant_min_bound,
            self.quant_round_type,
        )
        return out

    def compute_paddle_output(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x = paddle.to_tensor(self.x)
        bias = paddle.to_tensor(self.bias)
        dequant_scales = paddle.to_tensor(self.dequant_scales)
        shift = paddle.to_tensor(self.shift)
        smooth = paddle.to_tensor(self.smooth)

        out = fused_bias_act(
            x=x,
            bias=bias,
            dequant_scales=dequant_scales,
            shift=shift,
            smooth=smooth,
            act_method=self.act_method,
            compute_dtype=self.compute_dtype,
            quant_scale=self.quant_scale,
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
        )

        return out


class TestDequantFP32(TestQuantFP32):
    def init_test_case(self):
        self.rows = 10
        self.cols = 10
        self.atol = 1

        self.dtype = 'float32'
        self.compute_dtype = 'fp32'
        self.quant_scale = 0.5
        self.quant_round_type = 1
        self.quant_max_bound = 127.0
        self.quant_min_bound = -127.0

    def generate_inputs(self):
        self.x = np.random.randint(
            low=-16, high=16, size=(self.batch_size, self.seq_len, self.cols)
        ).astype('int32')
        self.bias = np.random.rand(self.cols).astype(self.dtype)
        self.dequant_scales = np.ones(self.cols).astype('float32')

    def compute_baseline_output(self):
        input_dequanted = fake_dequant(self.x, self.dequant_scales)
        out = gelu(input_dequanted + self.bias).astype(self.dtype)
        return out

    def compute_paddle_output(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x = paddle.to_tensor(self.x)
        bias = paddle.to_tensor(self.bias)
        dequant_scales = paddle.to_tensor(self.dequant_scales)

        out = fused_bias_act(
            x=x,
            bias=bias,
            dequant_scales=dequant_scales,
            act_method=self.act_method,
            compute_dtype=self.compute_dtype,
        )
        return out


class TestQuantFP16(TestQuantFP32):
    def init_test_case(self):
        self.atol = 1

        self.dtype = 'float16'
        self.compute_dtype = 'fp16'
        self.quant_scale = 0.5
        self.quant_round_type = 1
        self.quant_max_bound = 127.0
        self.quant_min_bound = -127.0


class TestDequantFP16(TestDequantFP32):
    def init_test_case(self):
        self.rows = 10
        self.cols = 10
        self.atol = 1

        self.dtype = 'float16'
        self.compute_dtype = 'fp16'
        self.quant_scale = 0.5
        self.quant_round_type = 1
        self.quant_max_bound = 127.0
        self.quant_min_bound = -127.0


class TestQuantGegluFP16(TestQuantFP32):
    def init_test_case(self):
        self.atol = 1

        self.dtype = 'float16'
        self.compute_dtype = 'fp16'
        self.act_method = 'geglu'
        self.quant_scale = 0.5
        self.quant_round_type = 1
        self.quant_max_bound = 127.0
        self.quant_min_bound = -127.0

        self.use_glu = True

    def compute_baseline_output(self):
        input_dequanted = fake_dequant(self.x, self.dequant_scales)
        tmp = (input_dequanted + self.bias).astype('float32')
        tmp_head = tmp[:, :, : self.cols // 2]
        tmp_tail = tmp[:, :, self.cols // 2 :]
        out_tmp = gelu(tmp_head).astype('float32') * tmp_tail

        out = fake_quant(
            out_tmp,
            self.shift,
            self.smooth,
            self.quant_scale,
            self.quant_max_bound,
            self.quant_min_bound,
            self.quant_round_type,
        )
        return out


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestFusedBiasActOpBF16(unittest.TestCase):
    def setUp(self):
        paddle.seed(2019)
        np.random.seed(2019)

        self.op_type = "fused_bias_act"
        self.rtol = 1e-3
        self.atol = 1e-3

        self.batch_size = 2
        self.seq_len = 20
        self.cols = 512

        self.act_method = 'gelu'
        self.compute_dtype = 'default'

        self.init_test_case()
        self.generate_inputs()

    def init_test_case(self):
        pass

    def generate_inputs(self):
        self.x = (
            np.random.rand(self.batch_size, self.seq_len, self.cols).astype(
                'float32'
            )
            * 16
        )
        self.bias = np.random.rand(self.cols).astype('float32')

    def compute_baseline_output(self):
        out = gelu(self.x.astype('float32') + self.bias)
        return convert_float_to_uint16(out)

    def compute_paddle_output(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x = paddle.to_tensor(convert_float_to_uint16(self.x))
        bias = paddle.to_tensor(convert_float_to_uint16(self.bias))

        out = fused_bias_act(
            x=x,
            bias=bias,
            act_method=self.act_method,
            compute_dtype=self.compute_dtype,
        )
        return out

    def test_check_output(self):
        final_out_ref = self.compute_baseline_output()
        final_out = self.compute_paddle_output()
        np.testing.assert_allclose(
            final_out_ref, final_out, rtol=self.rtol, atol=self.atol
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestWithComTypeBF16(unittest.TestCase):
    def init_test_case(self):
        self.act_method = 'geglu'
        self.compute_dtype = 'bf16'


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestGegluBF16(TestFusedBiasActOpBF16):
    def init_test_case(self):
        self.act_method = 'geglu'
        self.compute_dtype = 'default'

    def compute_baseline_output(self):
        res_tmp = self.x + self.bias
        res_tmp_head = res_tmp[:, :, : self.cols // 2]
        res_tmp_tail = res_tmp[:, :, self.cols // 2 :]
        res_tmp_head_act = gelu(res_tmp_head)
        out = res_tmp_head_act * res_tmp_tail
        return convert_float_to_uint16(out)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16 ",
)
class TestSwigluBF16(TestFusedBiasActOpBF16):
    def init_test_case(self):
        self.act_method = 'swiglu'
        self.compute_dtype = 'default'

    def compute_baseline_output(self):
        res_tmp = self.x + self.bias
        res_tmp_head = res_tmp[:, :, : self.cols // 2]
        res_tmp_tail = res_tmp[:, :, self.cols // 2 :]
        res_tmp_head_act = swish(res_tmp_head)
        out = res_tmp_head_act * res_tmp_tail
        return convert_float_to_uint16(out)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestQuantBF16(TestFusedBiasActOpBF16):
    def init_test_case(self):
        self.atol = 1

        self.compute_dtype = 'bf16'
        self.act_method = 'gelu'
        self.quant_scale = 0.5
        self.quant_round_type = 1
        self.quant_max_bound = 127.0
        self.quant_min_bound = -127.0

        self.use_glu = False

    def generate_inputs(self):
        self.x = np.random.randint(
            low=-1000,
            high=1000,
            size=(self.batch_size, self.seq_len, self.cols),
        ).astype('int32')
        self.bias = np.zeros(self.cols).astype('float32')
        self.dequant_scales = np.ones(self.cols).astype('float32')

        quant_params_cols = self.cols // 2 if self.use_glu else self.cols
        self.shift = np.zeros(quant_params_cols).astype('float32')
        self.smooth = np.ones(quant_params_cols).astype('float32')

    def compute_baseline_output(self):
        input_dequanted = fake_dequant(
            self.x.astype('float32'), self.dequant_scales
        )
        output_tmp = gelu(input_dequanted + self.bias)
        out = fake_quant(
            output_tmp,
            self.shift,
            self.smooth,
            self.quant_scale,
            self.quant_max_bound,
            self.quant_min_bound,
            self.quant_round_type,
        )

        return out

    def compute_paddle_output(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x = paddle.to_tensor(self.x)
        bias = paddle.to_tensor(convert_float_to_uint16(self.bias))
        dequant_scales = paddle.to_tensor(self.dequant_scales)
        shift = paddle.to_tensor(convert_float_to_uint16(self.shift))
        smooth = paddle.to_tensor(convert_float_to_uint16(self.smooth))

        out = fused_bias_act(
            x=x,
            bias=bias,
            dequant_scales=dequant_scales,
            shift=shift,
            smooth=smooth,
            act_method=self.act_method,
            compute_dtype=self.compute_dtype,
            quant_scale=self.quant_scale,
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
        )
        return out


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestQuantGegluBF16(TestQuantBF16):
    def init_test_case(self):
        self.atol = 1

        self.compute_dtype = 'bf16'
        self.act_method = 'geglu'
        self.quant_scale = 0.5
        self.quant_round_type = 1
        self.quant_max_bound = 127.0
        self.quant_min_bound = -127.0

        self.use_glu = True

    def compute_baseline_output(self):
        input_dequanted = fake_dequant(
            self.x.astype('float32'), self.dequant_scales
        )
        tmp = (input_dequanted + self.bias).astype('float32')
        tmp_head = tmp[:, :, : self.cols // 2]
        tmp_tail = tmp[:, :, self.cols // 2 :]
        out_tmp = gelu(tmp_head).astype('float32') * tmp_tail

        out = fake_quant(
            out_tmp,
            self.shift,
            self.smooth,
            self.quant_scale,
            self.quant_max_bound,
            self.quant_min_bound,
            self.quant_round_type,
        )

        return out


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestQuantSwigluBF16(TestQuantBF16):
    def init_test_case(self):
        self.atol = 1

        self.compute_dtype = 'bf16'
        self.act_method = 'swiglu'
        self.quant_scale = 0.5
        self.quant_round_type = 1
        self.quant_max_bound = 127.0
        self.quant_min_bound = -127.0

        self.use_glu = True

    def compute_baseline_output(self):
        input_dequanted = fake_dequant(
            self.x.astype('float32'), self.dequant_scales
        )
        tmp = (input_dequanted + self.bias).astype('float32')
        tmp_head = tmp[:, :, : self.cols // 2]
        tmp_tail = tmp[:, :, self.cols // 2 :]
        out_tmp = swish(tmp_head).astype('float32') * tmp_tail

        out = fake_quant(
            out_tmp,
            self.shift,
            self.smooth,
            self.quant_scale,
            self.quant_max_bound,
            self.quant_min_bound,
            self.quant_round_type,
        )

        return out


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestQuantSwigluFP8(TestQuantBF16):
    def init_test_case(self):
        self.atol = 1

        self.compute_dtype = 'bf16'
        self.act_method = 'swiglu'
        self.quant_scale = 0.5
        self.quant_round_type = 1
        self.quant_max_bound = 448.0
        self.quant_min_bound = -448.0

        self.use_glu = True

    def compute_baseline_output(self):
        input_dequanted = fake_dequant(
            self.x.astype('float32'), self.dequant_scales
        )
        tmp = (input_dequanted + self.bias).astype('float32')
        tmp_head = tmp[:, :, : self.cols // 2]
        tmp_tail = tmp[:, :, self.cols // 2 :]
        out_tmp = swish(tmp_head).astype('float32') * tmp_tail

        out = fake_quant(
            out_tmp,
            self.shift,
            self.smooth,
            self.quant_scale,
            self.quant_max_bound,
            self.quant_min_bound,
            self.quant_round_type,
        )

        return out


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_rocm(),
    "core is not compiled with CUDA or ROCm",
)
class TestAssert(unittest.TestCase):
    def setUp(self):
        self.rows = 20
        self.cols = 512

        self.dtype = 'float32'
        self.act_method = 'gelu'

    def test_assert_case1(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x = np.random.randint(
            low=-16, high=16, size=(self.rows, self.cols)
        ).astype('int32')

        bias = np.random.rand(self.cols).astype(self.dtype)

        try:
            out = fused_bias_act(
                x=paddle.to_tensor(x),
                bias=paddle.to_tensor(bias),
            )
        except ValueError as e:
            pass

    def test_assert_case2(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x = np.random.randint(
            low=-16, high=16, size=(self.rows, self.cols)
        ).astype('int32')

        bias = np.random.rand(self.cols).astype(self.dtype)

        try:
            out = fused_bias_act(
                x=paddle.to_tensor(x),
                bias=paddle.to_tensor(bias),
                compute_dtype='fp16',
            )
        except ValueError as e:
            pass

    def test_assert_case3(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x = np.random.randint(
            low=-16, high=16, size=(self.rows, self.cols)
        ).astype('int32')

        bias = np.random.rand(self.cols).astype(self.dtype)
        act_method = "error_type"
        try:
            out = fused_bias_act(
                x=paddle.to_tensor(x),
                bias=paddle.to_tensor(bias),
                compute_dtype='fp16',
                act_method=act_method,
            )
        except ValueError as e:
            pass


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_rocm(),
    "core is not compiled with CUDA or ROCm",
)
class TestWithoutBias(unittest.TestCase):
    def setUp(self):
        paddle.seed(2017)
        np.random.seed(2017)

        self.op_type = "fused_bias_act"
        self.rtol = 1e-5
        self.atol = 1e-3

        self.batch_size = 2
        self.seq_len = 20
        self.cols = 512

        self.dtype = 'float32'
        self.act_method = 'gelu'

        self.use_glu = False

        self.init_test_case()
        self.generate_inputs()

    def init_test_case(self):
        pass

    def generate_inputs(self):
        self.x = (
            np.random.rand(self.batch_size, self.seq_len, self.cols) * 16
        ).astype(self.dtype)
        # self.bias = np.random.rand(self.cols).astype(self.dtype)

    def compute_baseline_output(self):
        out = gelu(self.x).astype(self.dtype)
        return out

    def compute_paddle_output(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x = paddle.to_tensor(self.x)

        return fused_bias_act(
            x=x,
            bias=None,
            act_method=self.act_method,
        )

    def test_check_output(self):
        final_out_ref = self.compute_baseline_output()
        final_out = self.compute_paddle_output()
        np.testing.assert_allclose(
            final_out_ref, final_out, rtol=self.rtol, atol=self.atol
        )


if __name__ == '__main__':
    unittest.main()
