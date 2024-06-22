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
from paddle.base.layer_helper import LayerHelper
from paddle.base.framework import OpProtoHolder
from paddle.framework import in_dynamic_or_pir_mode
from paddle.incubate.tt import paddle_use_triton, tune_and_invoke_part2

triton_softmax_template = (
    """
std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor& x) {

  // 下面的变量必须和triton kernel里的参数名字一致!
  int M = x.shape()[0];
  int n_cols = x.shape()[1];

  int input_row_stride = n_cols;
  int output_row_stride = n_cols;

  auto out = paddle::empty(x.shape(), x.dtype(), x.place());
  auto output_ptr = get_tensor_ptr(out);
  auto input_ptr = get_tensor_ptr(x);

  auto run_stream = x.stream();
"""
    + tune_and_invoke_part2
    + """
  return {out};
}

std::vector<std::vector<int64_t>> ${op_name}_InferShape(const std::vector<int64_t>& A_shape) {
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
    {'num_warps': 1},
    {'num_warps': 2},
    {'num_warps': 4},
]

@paddle_use_triton(
    tune_config=softmax_kernel_config,
    custom_op_template=triton_softmax_template,
    key=["M"]
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
    op_name = "triton_softmax"
    # n_cols must be >0 even if when d2s
    assert type(n_cols) == int and n_cols > 0
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    op_name += str(BLOCK_SIZE)

    input_row_stride = n_cols
    output_row_stride = n_cols

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        # 这里的代码仅仅是注册这个Op！
        y = paddle.empty_like(x)
        # 这里需要注意grid传入的字符串必须是形参的运算得来的！
        grid = ("M",)
        softmax_kernel[(op_name, grid)](
            y,
            x,
            M,
            input_row_stride,
            output_row_stride,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # 上面是已经注册完这个Op了，下面开始真正的调用这个自定义算子啦。
    if in_dynamic_or_pir_mode():
        #print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(op_name, x)
        return outs[0]
    else:
        print(f"== we are in dynamic to static mode, op_name: {op_name}")
        helper = LayerHelper(op_name, **locals())
        inputs = {
            'x': x,
        }
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type=op_name,
            inputs=inputs,
            outputs={'out': out},
        )
        return out



# 这里封装了个 softmax_layer 类，是用来验证装饰器的！
class softmax_layer(paddle.nn.Layer):
    def __init__(self, hidd):
        super().__init__()
        self.fn = paddle.nn.Linear(hidd, hidd, bias_attr=False)
    
    @paddle.jit.to_static(backend="paddle_inference")
    def forward(self, x):
        for i in range(1000):
            #x = paddle.nn.functional.softmax(x,-1)
            x = softmax(x)
        x = x.cast("float32")
        x = self.fn(x)
        return x


batch = 4096
hidd = 1024
dtype = "bfloat16"
x = paddle.rand([batch, hidd], dtype=dtype)

# this is for inference decorator.
mylayer = softmax_layer(hidd)
mylayer(x)

# warm up.
for i in range(100):
    out = softmax(x)
    baseline = paddle.nn.functional.softmax(x, -1)

print(paddle.max(paddle.abs(out - baseline)))

import datetime
import time

repeat_times = 100
paddle.device.synchronize()

starttime = datetime.datetime.now()

for i in range(repeat_times):
    out = softmax(x)

paddle.device.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The triton whoel end to end time : ", time_ms, "ms")

paddle.device.synchronize()
starttime = datetime.datetime.now()

for i in range(repeat_times):
    baseline = paddle.nn.functional.softmax(x, -1)

paddle.device.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The paddle whoel end to end time : ", time_ms, "ms")

