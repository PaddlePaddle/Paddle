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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import triton
import triton.language as tl

import paddle
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.framework import in_dynamic_or_pir_mode
from paddle.incubate.tt import paddle_use_triton, tune_and_invoke_part2

triton_softmax_template = (
    """
std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor& x) {
  int M = x.shape()[0];
  int n_cols = x.shape()[1];

  int input_row_stride = n_cols;
  int output_row_stride = n_cols;

  auto out = paddle::empty({M, n_cols}, x.dtype(), x.place());

  auto input_ptr = get_tensor_ptr(x);
  auto output_ptr = get_tensor_ptr(out);

  auto run_stream = out.stream();

  std::vector<int> problem_size = {M,};
"""
    + tune_and_invoke_part2
    + """
  return {out};
}

std::vector<std::vector<int64_t>> ${op_name}_InferShape(
        const std::vector<int64_t>& A_shape) {
  return {A_shape};
}

std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
  return {A_dtype};
}

PD_BUILD_OP(${op_name})
    .Inputs({"x"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(${op_name}_func))
    .SetInferDtypeFn(PD_INFER_DTYPE(${op_name}_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(${op_name}_InferShape));
"""
)

softmax_kernel_config = [
    {'num_warps': 4},
    {'num_warps': 1},
    {'num_warps': 2},
]


@paddle_use_triton(
    tune_config=softmax_kernel_config,
    custom_op_template=triton_softmax_template,
)
def softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax(x):
    M, n_cols = x.shape
    op_name = "triron_softmax"
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    op_name += str(BLOCK_SIZE)

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        # 这里的代码仅仅是注册这个Op！
        y = paddle.empty_like(x)
        # 这里需要注意grid传入的字符串必须是形参的运算得来的！！！！
        grid = ("M",)
        softmax_kernel[(op_name, grid)](
            y,
            x,
            M,
            n_cols,
            n_cols,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # 上面是已经注册完这个Op了，下面开始真正的调用这个自定义算子啦。
    if in_dynamic_or_pir_mode():
        print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(op_name, x)
        return outs[0]
    else:
        # 这里做动转静，暂时省略。
        pass


batch = 4096
hidd = 4096
dtype = "float16"
x = paddle.rand([batch, hidd], dtype=dtype)

for i in range(100):
    out = softmax(x)

for i in range(100):
    baseline = paddle.nn.functional.softmax(x, -1)

print(paddle.max(paddle.abs(out - baseline)))
