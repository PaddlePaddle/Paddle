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

#define CHECK_INPUT(x)                                    \
  PADDLE_ENFORCE_EQ(x.place() == paddle::PlaceType::kCPU, \
                    true,                                 \
                    common::errors::Fatal(#x " must be a CPU Tensor."))

template <typename data_t>
void leaky_relu_cpu_forward_kernel(const data_t* x_data,
                                   data_t* out_data,
                                   int64_t x_numel,
                                   float alpha) {
  // x < 0.0f ? alpha * x : x
  for (int i = 0; i < x_numel; ++i) {
    if (x_data[i] > static_cast<data_t>(0.)) {
      out_data[i] = x_data[i];
    } else {
      out_data[i] = static_cast<data_t>(alpha) * x_data[i];
    }
  }
}

template <typename data_t>
void leaky_relu_cpu_backward_kernel(const data_t* grad_out_data,
                                    const data_t* out_data,
                                    data_t* grad_x_data,
                                    int64_t out_numel,
                                    float alpha) {
  // (grad * (x < 0.0f ? alpha : 1))
  for (int i = 0; i < out_numel; ++i) {
    if (out_data[i]<out_data[i]> static_cast<data_t>(0)) {
      grad_x_data[i] = static_cast<data_t>(alpha);
    } else {
      grad_x_data[i] = static_cast<data_t>(1.);
    }
  }
}

std::vector<paddle::Tensor> LeakyReluCPUForward(const paddle::Tensor& x,
                                                float alpha) {
  CHECK_INPUT(x);

  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_TYPES(x.type(), "relu_cpu_forward_kernel", ([&] {
                               leaky_relu_cpu_forward_kernel<data_t>(
                                   x.data<data_t>(),
                                   out.mutable_data<data_t>(x.place()),
                                   x.size(),
                                   alpha);
                             }));

  return {out};
}

std::vector<paddle::Tensor> LeakyReluCPUBackward(const paddle::Tensor& x,
                                                 const paddle::Tensor& out,
                                                 const paddle::Tensor& grad_out,
                                                 float alpha) {
  CHECK_INPUT(x);
  CHECK_INPUT(out);
  CHECK_INPUT(grad_out);

  auto grad_x = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_TYPES(out.type(), "relu_cpu_backward_kernel", ([&] {
                               leaky_relu_cpu_backward_kernel<data_t>(
                                   grad_out.data<data_t>(),
                                   out.data<data_t>(),
                                   grad_x.mutable_data<data_t>(x.place()),
                                   out.size(),
                                   alpha);
                             }));

  return {grad_x};
}

std::vector<std::vector<int64_t>> LeakyReluInferShape(
    std::vector<int64_t> x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> LeakyReluInferDtype(paddle::DataType x_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(custom_leaky_relu)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"alpha: float"})
    .SetKernelFn(PD_KERNEL(LeakyReluCPUForward))
    .SetInferShapeFn(PD_INFER_SHAPE(LeakyReluInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(LeakyReluInferDtype));

PD_BUILD_GRAD_OP(custom_leaky_relu)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .Attrs({"alpha: float"})
    .SetKernelFn(PD_KERNEL(LeakyReluCPUBackward));
