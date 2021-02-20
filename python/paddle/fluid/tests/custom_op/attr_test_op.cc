// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

template <typename data_t>
void assign_cpu_kernel(const data_t* x_data,
                       data_t* out_data,
                       int64_t x_numel) {
  for (int i = 0; i < x_numel; ++i) {
    out_data[i] = x_data[i];
  }
}

std::vector<paddle::Tensor> AttrTest(const paddle::Tensor& x, int int_attr) {
  auto out = paddle::Tensor(paddle::PlaceType::kCPU);
  out.reshape(x.shape());

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "assign_cpu_kernel", ([&] {
        assign_cpu_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(), x.size());
      }));

  std::cout << std::endl;
  std::cout << int_attr;

  return {out};
}

std::vector<std::vector<int64_t>> InferShape(std::vector<int64_t> x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> InferDType(paddle::DataType x_dtype) {
  return {x_dtype};
}

PD_BUILD_OP("attr_test")
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"int_attr: int"})
    .SetKernelFn(PD_KERNEL(AttrTest))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDType));
