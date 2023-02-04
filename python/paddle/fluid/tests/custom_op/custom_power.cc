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

#include "paddle/extension.h"

std::vector<paddle::Tensor> PowerForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x * x};
  } else {
    PD_THROW("Not implemented.");
  }
}

std::vector<paddle::Tensor> PowerBackward(const paddle::Tensor& x,
                                          const paddle::Tensor& out,
                                          const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    paddle::Tensor middle_result = grad_out * x;
    return {paddle::add(middle_result, middle_result)};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_power)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(PowerForward));

PD_BUILD_GRAD_OP(custom_power)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(PowerBackward));
