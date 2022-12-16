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

#include <cmath>
#include <iostream>
#include <vector>

#include "paddle/extension.h"

#define CHECK_CPU_INPUT(x) \
  PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

template <typename data_t>
void tanh_cpu_forward_kernel(const data_t* x_data,
                             data_t* out_data,
                             int64_t x_numel) {
  PD_CHECK(x_data != nullptr, "x_data is nullptr.");
  PD_CHECK(out_data != nullptr, "out_data is nullptr.");
  for (int64_t i = 0; i < x_numel; ++i) {
    out_data[i] = std::tanh(x_data[i]);
  }
}

template <typename data_t>
void tanh_cpu_backward_kernel(const data_t* grad_out_data,
                              const data_t* out_data,
                              data_t* grad_x_data,
                              int64_t out_numel) {
  PD_CHECK(grad_out_data != nullptr, "grad_out_data is nullptr.");
  PD_CHECK(out_data != nullptr, "out_data is nullptr.");
  PD_CHECK(grad_x_data != nullptr, "grad_x_data is nullptr.");
  for (int64_t i = 0; i < out_numel; ++i) {
    grad_x_data[i] =
        grad_out_data[i] * (static_cast<data_t>(1) - out_data[i] * out_data[i]);
  }
}

template <typename data_t>
void tanh_cpu_double_backward_kernel(const data_t* out_data,
                                     const data_t* ddx_data,
                                     const data_t* dout_data,
                                     data_t* dout_new_data,
                                     data_t* ddout_data,
                                     int64_t ddout_numel) {
  PD_CHECK(out_data != nullptr, "out_data is nullptr.");
  PD_CHECK(ddx_data != nullptr, "ddx_data is nullptr.");
  PD_CHECK(dout_data != nullptr, "dout_data is nullptr.");
  PD_CHECK(dout_new_data != nullptr, "dout_new_data is nullptr.");
  PD_CHECK(ddout_data != nullptr, "ddout_data is nullptr.");
  for (int64_t i = 0; i < ddout_numel; ++i) {
    dout_new_data[i] = static_cast<data_t>(-1) * dout_data[i] *
                       static_cast<data_t>(2) * out_data[i] * ddx_data[i];
    ddout_data[i] =
        ddx_data[i] * (static_cast<data_t>(1) - out_data[i] * out_data[i]);
  }
}

std::vector<paddle::Tensor> TanhForward(const paddle::Tensor& x) {
  CHECK_CPU_INPUT(x);
  auto out = paddle::empty(x.shape(), x.dtype(), x.place());

  PD_DISPATCH_FLOATING_TYPES(
      x.dtype(), "tanh_cpu_forward", ([&] {
        tanh_cpu_forward_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(x.place()), x.size());
      }));

  return {out};
}

std::vector<paddle::Tensor> TanhBackward(const paddle::Tensor& out,
                                         const paddle::Tensor& grad_out) {
  CHECK_CPU_INPUT(out);
  auto grad_x = paddle::empty(out.shape(), out.dtype(), out.place());

  PD_DISPATCH_FLOATING_TYPES(out.dtype(), "tanh_cpu_backward", ([&] {
                               tanh_cpu_backward_kernel<data_t>(
                                   grad_out.data<data_t>(),
                                   out.data<data_t>(),
                                   grad_x.mutable_data<data_t>(out.place()),
                                   out.size());
                             }));

  return {grad_x};
}

std::vector<paddle::Tensor> TanhDoubleBackward(const paddle::Tensor& out,
                                               const paddle::Tensor& ddx,
                                               const paddle::Tensor& dout) {
  CHECK_CPU_INPUT(out);
  CHECK_CPU_INPUT(ddx);
  CHECK_CPU_INPUT(dout);
  auto dout_new = paddle::empty(out.shape(), out.dtype(), out.place());
  auto ddout = paddle::empty(out.shape(), out.dtype(), out.place());

  PD_DISPATCH_FLOATING_TYPES(out.dtype(), "tanh_cpu_double_backward", ([&] {
                               tanh_cpu_double_backward_kernel<data_t>(
                                   out.data<data_t>(),
                                   ddx.data<data_t>(),
                                   dout.data<data_t>(),
                                   dout_new.mutable_data<data_t>(out.place()),
                                   ddout.mutable_data<data_t>(out.place()),
                                   ddout.size());
                             }));

  return {dout_new, ddout};
}

std::vector<std::vector<int64_t>> TanhBackwardInferShape(
    const std::vector<int64_t>& out_shape,
    const std::vector<int64_t>& dout_shape) {
  return {out_shape};
}

std::vector<std::vector<int64_t>> TanhDoubleBackwardInferShape(
    const std::vector<int64_t>& out_shape,
    const std::vector<int64_t>& ddx_shape,
    const std::vector<int64_t>& dout_shape) {
  return {dout_shape, dout_shape};
}

PD_BUILD_OP(custom_tanh)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(TanhForward));

PD_BUILD_GRAD_OP(custom_tanh)
    .Inputs({"Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(TanhBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(TanhBackwardInferShape));

PD_BUILD_DOUBLE_GRAD_OP(custom_tanh)
    .Inputs({"Out", paddle::Grad(paddle::Grad("X")), paddle::Grad("Out")})
    .Outputs({paddle::New(paddle::Grad("Out")),
              paddle::Grad(paddle::Grad("Out"))})
    .SetKernelFn(PD_KERNEL(TanhDoubleBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(TanhDoubleBackwardInferShape));
