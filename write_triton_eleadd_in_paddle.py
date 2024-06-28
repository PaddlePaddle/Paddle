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

import triton.language as tl

import paddle
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode
from paddle.incubate.tt import paddle_use_triton, tune_and_invoke_part2

triton_add_template = (
    """
std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor& x, const paddle::Tensor& y) {

  // 下面的变量必须和triton kernel里的参数名字一致!
  int n_elements = 1;
  for (int i = 0; i < x.shape().size(); ++i) {
    n_elements *= x.shape()[i];
  }

  auto out = paddle::empty(x.shape(), x.dtype(), x.place());
  auto x_ptr = get_tensor_ptr(x);
  auto y_ptr = get_tensor_ptr(y);
  auto output_ptr = get_tensor_ptr(out);

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
    .Inputs({"x", "y"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(${op_name}_func))
    .SetInferDtypeFn(PD_INFER_DTYPE(${op_name}_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(${op_name}_InferShape));
"""
)


@paddle_use_triton(custom_op_template=triton_add_template, key=["n_elements"])
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x, y):
    n_elements = x.shape[0] * x.shape[1]
    op_name = "triton_add"
    add_kernel_config = [
        {'num_warps': 1},
    ]
    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        # 这里的代码仅仅是注册这个Op！
        out = paddle.empty_like(x)
        # 这里需要注意grid传入的字符串必须是形参的运算得来的！
        grid = ("(n_elements+BLOCK_SIZE-1)/BLOCK_SIZE",)
        add_kernel[(op_name, grid, add_kernel_config)](
            x,
            y,
            out,
            -1,  # n_elements
            BLOCK_SIZE=2048,
        )

    # 上面是已经注册完这个Op了，下面开始真正的调用这个自定义算子啦。
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(op_name, x, y)
        return outs[0]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            'x': x,
            'y': y,
        }
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type=op_name,
            inputs=inputs,
            outputs={'out': out},
        )
        return out


# 这里封装了个 add_layer 类，是用来验证装饰器的！
class add_layer(paddle.nn.Layer):
    def __init__(self, hidd):
        super().__init__()
        self.fn = paddle.nn.Linear(hidd, hidd, bias_attr=False)

    @paddle.jit.to_static(backend="paddle_inference")
    def forward(self, x, y):
        for i in range(1000):
            x = add(x, y)
        x = x.cast("float32")
        x = self.fn(x)
        return x


batch = 4096
hidd = 4096
dtype = "float32"
x = paddle.rand([batch, hidd], dtype=dtype)
y = paddle.rand([batch, hidd], dtype=dtype)

# warm up.
for i in range(100):
    out = add(x, y)
    baseline = paddle.add(x, y)

print(paddle.max(paddle.abs(out - baseline)))
