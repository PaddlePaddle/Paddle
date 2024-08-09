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

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import triton
import triton.language as tl

import paddle
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode
from paddle.incubate.tt import (
    get_dtype_str,
    paddle_use_triton,
    tune_and_invoke_part,
)

triton_modulate_template = (
    """

std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor &x,
    const paddle::Tensor &scale,
    const paddle::Tensor &shift) {
  int M = x.dims()[0] * x.dims()[1];
  int N = x.dims()[2];
  int seq_size = x.dims()[1];
  auto y = paddle::empty(x.shape(), x.dtype(), x.place());

  auto x_ptr = get_tensor_ptr(x);
  auto y_ptr = get_tensor_ptr(y);
  auto scale_ptr = get_tensor_ptr(scale);
  auto shift_ptr = get_tensor_ptr(shift);
  auto run_stream = y.stream();
"""
    + tune_and_invoke_part
    + """
  return {y};
}

std::vector<std::vector<int64_t>> ${op_name}_InferShape(
        const std::vector<int64_t>& A_shape) {
  return {A_shape};
}

std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype};
}

PD_BUILD_OP(${op_name})
    .Inputs({"x", "scale", "shift"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(${op_name}_func))
    .SetInferDtypeFn(PD_INFER_DTYPE(${op_name}_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(${op_name}_InferShape));
"""
)


@paddle_use_triton(
    custom_op_template=triton_modulate_template,
    key=["M"],
)
def modulate_kernel(
    x_ptr,
    y_ptr,
    scale_ptr,
    shift_ptr,
    M,
    N,
    seq_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    x_ptr += row * N
    y_ptr += row * N

    scale_ptr += (row // seq_size) * N
    shift_ptr += (row // seq_size) * N
    for col_off in range(0, N, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_hat = tl.load(x_ptr + cols, mask=mask, other=0.0)

        scales = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        shifts = tl.load(shift_ptr + cols, mask=mask, other=0.0)
        my_scale = 1 + scales
        my_scale = my_scale.to(shifts.dtype)
        y = x_hat * (my_scale) + shifts
        tl.store(y_ptr + cols, y, mask=mask)


def triton_modulate(x, scale, shift):
    assert (
        len(x.shape) == 3
    ), "x should be 3-dim [batch_size, seq_size, feature_dim]"

    assert (
        len(scale.shape) == 2 and len(shift.shape) == 2
    ), "scale and shift should be 2-dim [batch_size, feature_dim]"
    assert (
        scale.shape[0] == shift.shape[0] == x.shape[0]
    ), "x, scale and shift should have same shape[0] == batch_size"
    assert (
        scale.shape[1] == shift.shape[1] == x.shape[-1]
    ), "x, scale and shift should have same shape[-1] == feature_dim"

    M = x.shape[0] * x.shape[1]
    N = x.shape[2]
    seq_size = x.shape[1]
    BLOCK_SIZE = min(4096, triton.next_power_of_2(N))

    op_name = "triton_modulate"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_{BLOCK_SIZE}"

    modulate_kernel_config = [
        {'num_warps': 1},
    ]

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        y = paddle.empty_like(x)
        grid = ("M",)
        modulate_kernel[(op_name, grid)](
            x,
            y,
            scale,
            shift,
            M,
            N,
            seq_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return y

    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(op_name, x, scale, shift)
        return outs[0]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            'x': x,
            'scale': scale,
            'shift': shift,
        }
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type=op_name,
            inputs=inputs,
            outputs={'out': out},
        )
        return out


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(axis=1)) + shift.unsqueeze(axis=1)


batch = 2
seq = 3600
hidd = 4096
dtype = "bfloat16"

x = paddle.rand([batch, seq, hidd], dtype=dtype)

shift_msa_x = paddle.rand([batch, hidd], dtype=dtype)
scale_msa_x = paddle.rand([batch, hidd], dtype=dtype)

for i in range(100):
    mt_result = triton_modulate(x, scale_msa_x, shift_msa_x)
    baseline = modulate(x, shift_msa_x, scale_msa_x)

print(paddle.max(paddle.abs(baseline - mt_result)))
