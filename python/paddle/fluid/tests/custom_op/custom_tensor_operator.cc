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

// y = x + 1
std::vector<paddle::Tensor> AddForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    paddle::Tensor ones = paddle::full(x.shape(), 1.0, x.dtype(), x.place());
    return {x + ones};
  } else {
    PD_THROW("Not implemented.");
  }
}

// dy / dx = 1 * grad_out
std::vector<paddle::Tensor> AddBackward(const paddle::Tensor& x,
                                        const paddle::Tensor& out,
                                        const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    paddle::Tensor ones = paddle::full(x.shape(), 1.0, x.dtype(), x.place());
    return {grad_out * ones};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_add)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AddForward));

PD_BUILD_GRAD_OP(custom_add)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(AddBackward));

// y = x + 1
std::vector<paddle::Tensor> ScalarAddForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x + 1};
  } else {
    PD_THROW("Not implemented.");
  }
}

// dy / dx = 1 * grad_out
std::vector<paddle::Tensor> ScalarAddBackward(const paddle::Tensor& x,
                                              const paddle::Tensor& out,
                                              const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    return {grad_out * 1};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_scalar_add)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ScalarAddForward));

PD_BUILD_GRAD_OP(custom_scalar_add)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(ScalarAddBackward));

// y = x - 1
std::vector<paddle::Tensor> SubtractForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    paddle::Tensor ones = paddle::full(x.shape(), 1, x.dtype(), x.place());
    return {x - ones};
  } else {
    PD_THROW("Not implemented.");
  }
}

// dy / dx = 1 * grad_out
std::vector<paddle::Tensor> SubtractBackward(const paddle::Tensor& x,
                                             const paddle::Tensor& out,
                                             const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    return {grad_out};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_subtract)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(SubtractForward));

PD_BUILD_GRAD_OP(custom_subtract)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(SubtractBackward));

// y = x - 1
std::vector<paddle::Tensor> ScalarSubtractForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x - 1};
  } else {
    PD_THROW("Not implemented.");
  }
}

// dy / dx = 1 * grad_out
std::vector<paddle::Tensor> ScalarSubtractBackward(
    const paddle::Tensor& x,
    const paddle::Tensor& out,
    const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    return {grad_out * 1};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_scalar_subtract)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ScalarSubtractForward));

PD_BUILD_GRAD_OP(custom_scalar_subtract)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(ScalarSubtractBackward));

// y = x * 5
std::vector<paddle::Tensor> MultiplyForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    paddle::Tensor ones = paddle::full(x.shape(), 1.0, x.dtype(), x.place());
    paddle::Tensor fives = paddle::experimental::fill(ones, 5);
    return {x * fives};
  } else {
    PD_THROW("Not implemented.");
  }
}

// dy / dx = 5 * grad_out
std::vector<paddle::Tensor> MultiplyBackward(const paddle::Tensor& x,
                                             const paddle::Tensor& out,
                                             const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    paddle::Tensor ones = paddle::full(x.shape(), 1.0, x.dtype(), x.place());
    paddle::Tensor fives = paddle::experimental::fill(ones, 5);
    return {fives * grad_out};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_multiply)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(MultiplyForward));

PD_BUILD_GRAD_OP(custom_multiply)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(MultiplyBackward));

// y = x * 5
std::vector<paddle::Tensor> ScalarMultiplyForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x * 5};
  } else {
    PD_THROW("Not implemented.");
  }
}

// dy / dx = grad_out * 5
std::vector<paddle::Tensor> ScalarMultiplyBackward(
    const paddle::Tensor& x,
    const paddle::Tensor& out,
    const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    return {grad_out * 5};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_scalar_multiply)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ScalarMultiplyForward));

PD_BUILD_GRAD_OP(custom_scalar_multiply)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(ScalarMultiplyBackward));

// y = 1 / x
std::vector<paddle::Tensor> DivideForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    paddle::Tensor ones = paddle::full(x.shape(), 1.0, x.dtype(), x.place());
    return {ones / x};
  } else {
    PD_THROW("Not implemented.");
  }
}

// dy / dx = - (1 / x / x) * grad_out
std::vector<paddle::Tensor> DivideBackward(const paddle::Tensor& x,
                                           const paddle::Tensor& out,
                                           const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    paddle::Tensor zeros = paddle::full(x.shape(), 0.0, x.dtype(), x.place());
    return {zeros - grad_out / (x * x)};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_divide)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(DivideForward));

PD_BUILD_GRAD_OP(custom_divide)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(DivideBackward));

// y = 1 / x / 1
std::vector<paddle::Tensor> ScalarDivideForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    paddle::Tensor ones = paddle::full(x.shape(), 1.0, x.dtype(), x.place());
    return {ones / x / 1};
  } else {
    PD_THROW("Not implemented.");
  }
}

// dy / dx = - (1 / x / x) * grad_out
std::vector<paddle::Tensor> ScalarDivideBackward(
    const paddle::Tensor& x,
    const paddle::Tensor& out,
    const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    paddle::Tensor zeros = paddle::full(x.shape(), 0.0, x.dtype(), x.place());
    return {zeros - grad_out / (x * x)};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_scalar_divide)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ScalarDivideForward));

PD_BUILD_GRAD_OP(custom_scalar_divide)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(ScalarDivideBackward));
