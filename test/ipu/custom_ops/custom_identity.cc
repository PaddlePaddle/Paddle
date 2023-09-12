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
std::vector<std::vector<int64_t>> InferShape(std::vector<int64_t> x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> InferDtype(paddle::DataType x_dtype) {
  return {x_dtype};
}

std::vector<paddle::Tensor> OpForward(const paddle::Tensor &x) { return {x}; }

std::vector<paddle::Tensor> OpBackward(const paddle::Tensor &x) { return {x}; }
}  // namespace

// https://github.com/graphcore/popart/blob/sdk-release-2.5/willow/src/builder.gen.cpp#L620
PD_BUILD_OP(custom_identity)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype))
    .SetKernelFn(PD_KERNEL(OpForward));

PD_BUILD_GRAD_OP(custom_identity)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(OpBackward));
