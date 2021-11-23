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

std::vector<paddle::Tensor> DispatchTestInterger(const paddle::Tensor& x) {
  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_INTEGRAL_TYPES(
      x.type(), "assign_cpu_kernel", ([&] {
        assign_cpu_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(), x.size());
      }));

  return {out};
}

PD_BUILD_OP(dispatch_test_integer)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(DispatchTestInterger));

std::vector<paddle::Tensor> DispatchTestFloatAndInteger(
    const paddle::Tensor& x) {
  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(
      x.type(), "assign_cpu_kernel", ([&] {
        assign_cpu_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(), x.size());
      }));

  return {out};
}

PD_BUILD_OP(dispatch_test_float_and_integer)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(DispatchTestFloatAndInteger));

std::vector<paddle::Tensor> DispatchTestComplex(const paddle::Tensor& x) {
  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_COMPLEX_TYPES(
      x.type(), "assign_cpu_kernel", ([&] {
        assign_cpu_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(), x.size());
      }));

  return {out};
}

PD_BUILD_OP(dispatch_test_complex)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(DispatchTestComplex));

std::vector<paddle::Tensor> DispatchTestFloatAndComplex(
    const paddle::Tensor& x) {
  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      x.type(), "assign_cpu_kernel", ([&] {
        assign_cpu_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(), x.size());
      }));

  return {out};
}

PD_BUILD_OP(dispatch_test_float_and_complex)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(DispatchTestFloatAndComplex));

std::vector<paddle::Tensor> DispatchTestFloatAndIntegerAndComplex(
    const paddle::Tensor& x) {
  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_AND_INTEGRAL_AND_COMPLEX_TYPES(
      x.type(), "assign_cpu_kernel", ([&] {
        assign_cpu_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(), x.size());
      }));

  return {out};
}

PD_BUILD_OP(dispatch_test_float_and_integer_and_complex)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(DispatchTestFloatAndIntegerAndComplex));

std::vector<paddle::Tensor> DispatchTestFloatAndHalf(const paddle::Tensor& x) {
  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_AND_HALF_TYPES(
      x.type(), "assign_cpu_kernel", ([&] {
        assign_cpu_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(), x.size());
      }));

  return {out};
}

PD_BUILD_OP(dispatch_test_float_and_half)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(DispatchTestFloatAndHalf));
