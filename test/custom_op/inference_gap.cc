// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <vector>
#include "NvInfer.h"

#include "paddle/extension.h"

void call_kernel(dim3 gridSize,
                 dim3 blockSize,
                 size_t share_M,
                 const cudaStream_t& stream,
                 const float* input,
                 float* output,
                 const int h,
                 const int w);

std::vector<paddle::Tensor> paddle_gap_forward(
    const paddle::Tensor& x,
    const std::vector<int>& test_attr1,
    const int test_attr2) {
  std::vector<int64_t> dims = x.shape();
  int32_t batch = dims[0];
  int32_t ch = dims[1];
  int32_t h = dims[2];
  int32_t w = dims[3];
  std::vector<int64_t> out_dims{batch, ch, test_attr2, test_attr2};
  auto out = paddle::empty(out_dims, x.dtype(), x.place());
  dim3 blockSize(ch);
  dim3 gridSize(batch);
  if (x.is_gpu() && x.dtype() == paddle::DataType::FLOAT32) {
    const float* input = x.data<float>();
    float* output = out.data<float>();
    PD_DISPATCH_FLOATING_TYPES(
        x.type(), "globalAvgPool", ([&] {
          call_kernel(gridSize, blockSize, 0, x.stream(), input, output, h, w);
        }));
  }
  return {out};
}

std::vector<std::vector<int64_t>> InferShape(std::vector<int64_t> x_shape,
                                             const std::vector<int>& test_attr1,
                                             const int test_attr2) {
  std::vector<int64_t> out_dims{x_shape[0], x_shape[1], test_attr2, test_attr2};
  return {out_dims};
}

std::vector<paddle::DataType> InferDtype(paddle::DataType x_dtype) {
  return {x_dtype};
}

nvinfer1::DimsExprs getOutputDimensions(
    std::pair<int32_t, int32_t> outputIndex_nbInputs,
    const nvinfer1::DimsExprs* inputs,
    nvinfer1::IExprBuilder& exprBuilder,  // NOLINT
    const std::vector<int>& test_attr1,
    const int test_attr2) noexcept {
  nvinfer1::DimsExprs dimsOutput(inputs[0]);
  dimsOutput.d[dimsOutput.nbDims - 1] = exprBuilder.constant(test_attr2);
  dimsOutput.d[dimsOutput.nbDims - 2] = exprBuilder.constant(test_attr2);
  return dimsOutput;
}

PD_BUILD_OP(gap)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(paddle_gap_forward))
    .Attrs({"test_attr1: std::vector<int>", "test_attr2: int"})
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype))
    .SetTrtInferShapeFn(PD_TRT_INFER_SHAPE(getOutputDimensions))
    .SetTrtSupportsFormatConfig({"float32:LINEAR+float32:LINEAR"});
