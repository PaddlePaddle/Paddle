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

template <typename data_t>
void fill_constant_cpu_kernel(data_t* out_data, int64_t x_numel, data_t value) {
  for (int i = 0; i < x_numel; ++i) {
    out_data[i] = value;
  }
}

std::vector<paddle::Tensor> MultiOutCPU(const paddle::Tensor& x) {
  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "assign_cpu_kernel", ([&] {
        assign_cpu_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(x.place()), x.size());
      }));

  // fake multi output: Fake_float64 with float64 dtype
  auto fake_float64 = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  fill_constant_cpu_kernel<double>(
      fake_float64.mutable_data<double>(x.place()), x.size(), 0.);

  // fake multi output: ZFake_int32 with int32 dtype
  auto zfake_int32 = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  fill_constant_cpu_kernel<int32_t>(
      zfake_int32.mutable_data<int32_t>(x.place()), x.size(), 1);

  return {out, fake_float64, zfake_int32};
}

std::vector<std::vector<int64_t>> InferShape(std::vector<int64_t> x_shape) {
  return {x_shape, x_shape, x_shape};
}

std::vector<paddle::DataType> InferDtype(paddle::DataType x_dtype) {
  return {x_dtype, paddle::DataType::FLOAT64, paddle::DataType::INT32};
}

PD_BUILD_OP(multi_out)
    .Inputs({"X"})
    .Outputs({"Out", "Fake_float64", "ZFake_int32"})
    .SetKernelFn(PD_KERNEL(MultiOutCPU))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype));
