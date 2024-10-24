// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdlib>
#include <iostream>
#include <vector>

#include "paddle/extension.h"

template <typename data_t>
void assign_cpu_kernel(const data_t* x_data,
                       data_t* out_data,
                       int64_t x_numel) {
  for (int i = 0; i < x_numel; ++i) {
    out_data[i] = x_data[i];
  }
}

void CheckAllForwardAttrs(const double& double_attr) {
  if (std::abs(double_attr - 3.14) > 1e-10) {
    throw std::runtime_error("double_attr value error.");
  }
}

std::vector<paddle::Tensor> ExtentAttrTestForward(const paddle::Tensor& x,
                                                  double double_attr) {
  auto out = paddle::empty_like(x);

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "assign_cpu_kernel", ([&] {
        assign_cpu_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(), x.size());
      }));

  // Check attrs value
  CheckAllForwardAttrs(double_attr);

  return {out};
}

std::vector<std::vector<int64_t>> ExtentAttrTestInferShape(
    const std::vector<int64_t>& x_shape, double double_attr) {
  return {x_shape};
}

PD_BUILD_OP(extend_attr_test)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"double_attr: double"})
    .SetKernelFn(PD_KERNEL(ExtentAttrTestForward))
    .SetInferShapeFn(PD_INFER_SHAPE(ExtentAttrTestInferShape));
