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

namespace {
std::vector<std::vector<int64_t>> InferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> y_shape,
    const std::string &reduction,
    const int &ignoreIndex,
    const bool &inputIsLogProbability) {
  // reduction type: Sum, Mean, None
  if (reduction == "None") {
    return {y_shape};
  } else {
    return {{1}};
  }
}

std::vector<paddle::DataType> InferDtype(paddle::DataType x_dtype,
                                         paddle::DataType y_dtype) {
  return {x_dtype};
}

std::vector<paddle::Tensor> OpForward(const paddle::Tensor &x,
                                      const paddle::Tensor &y) {
  return {x};
}

std::vector<paddle::Tensor> OpBackward(const paddle::Tensor &x) { return {x}; }
}  // namespace

// https://github.com/graphcore/popart/blob/sdk-release-2.5/willow/src/builder.cpp#L775
// type of `reduction` is std::string
// `ignoreIndex` is optional, if no need, need to remove it manually(which is a
// new custom op in paddle)
PD_BUILD_OP(custom_nll)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .Attrs({"reduction: std::string",
            "ignoreIndex: int",
            "inputIsLogProbability: bool"})
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype))
    .SetKernelFn(PD_KERNEL(OpForward));

PD_BUILD_GRAD_OP(custom_nll)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(OpBackward));
