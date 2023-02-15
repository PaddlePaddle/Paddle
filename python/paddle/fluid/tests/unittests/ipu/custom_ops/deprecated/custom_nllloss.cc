// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

std::vector<paddle::Tensor> Kernel_Function() { return {}; }
std::vector<paddle::Tensor> Kernel_Function_Grad() { return {}; }

// nllloss
std::vector<std::vector<int64_t>> InferShape_NllLoss(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> y_shape,
    const int& reduction,
    const std::string& ignoreIndex,
    const bool& inputIsLogProbability) {
  // 0: sum, 1: mean, 2: none
  if (reduction == 2) {
    return {y_shape};
  } else {
    return {{1}};
  }
}

std::vector<paddle::DataType> InferDtype_NllLoss(paddle::DataType x_dtype,
                                                 paddle::DataType y_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(custom_nll_loss)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .Attrs({"reduction: int",
            "ignoreIndex: std::string",
            "inputIsLogProbability: bool"})
    .SetKernelFn(PD_KERNEL(Kernel_Function))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape_NllLoss))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype_NllLoss));

PD_BUILD_GRAD_OP(custom_nll_loss)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(Kernel_Function_Grad));
