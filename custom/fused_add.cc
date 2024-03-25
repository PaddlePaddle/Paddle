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

#include "paddle/extension.h"

#include <vector>  // NOLINT

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

std::vector<paddle::Tensor> fused_add_forward(const paddle::Tensor& x,
                                              const paddle::Tensor& y);
std::vector<paddle::Tensor> fused_add_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& y,
                                               const paddle::Tensor& out,
                                               const paddle::Tensor& out_grad);

std::vector<paddle::Tensor> FusedAddForward(const paddle::Tensor& x,
                                            const paddle::Tensor& y) {
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  return fused_add_forward(x, y);
}

std::vector<paddle::Tensor> FusedAddBackward(const paddle::Tensor& x,
                                             const paddle::Tensor& y,
                                             const paddle::Tensor& out,
                                             const paddle::Tensor& out_grad) {
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  CHECK_INPUT(out);
  CHECK_INPUT(out_grad);
  return fused_add_backward(x, y, out, out_grad);
}

std::vector<std::vector<int64_t>> FusedAddInferShape(
    const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> FusedAddInferDtype(
    const paddle::DataType& x_dtype, const paddle::DataType& y_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(custom_fused_add)
    .Inputs({"X", "Y"})
    .Outputs({"OUT"})
    .SetKernelFn(PD_KERNEL(FusedAddForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedAddInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedAddInferDtype));

PD_BUILD_GRAD_OP(custom_fused_add)
    .Inputs({"X", "Y", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X"), paddle::Grad("Y")})
    .SetKernelFn(PD_KERNEL(FusedAddBackward));
