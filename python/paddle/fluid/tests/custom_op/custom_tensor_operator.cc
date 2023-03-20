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

// y = 1 + x
std::vector<paddle::Tensor> LeftScalarAddForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    return {1 + x};
  } else {
    PD_THROW("Not implemented.");
  }
}

// dy / dx = 1 * grad_out
std::vector<paddle::Tensor> LeftScalarAddBackward(
    const paddle::Tensor& x,
    const paddle::Tensor& out,
    const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    return {1 * grad_out};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_left_scalar_add)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(LeftScalarAddForward));

PD_BUILD_GRAD_OP(custom_left_scalar_add)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(LeftScalarAddBackward));

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

// y = - 1 + x
std::vector<paddle::Tensor> LeftScalarSubtractForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    return {-1 + x};
  } else {
    PD_THROW("Not implemented.");
  }
}

// dy / dx = 1 * grad_out
std::vector<paddle::Tensor> LeftScalarSubtractBackward(
    const paddle::Tensor& x,
    const paddle::Tensor& out,
    const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    return {1 * grad_out};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_left_scalar_subtract)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(LeftScalarSubtractForward));

PD_BUILD_GRAD_OP(custom_left_scalar_subtract)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(LeftScalarSubtractBackward));

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

// y = 5 * x
std::vector<paddle::Tensor> LeftScalarMultiplyForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    return {5 * x};
  } else {
    PD_THROW("Not implemented.");
  }
}

// dy / dx = 5 * grad_out
std::vector<paddle::Tensor> LeftScalarMultiplyBackward(
    const paddle::Tensor& x,
    const paddle::Tensor& out,
    const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    return {5 * grad_out};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_left_scalar_multiply)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(LeftScalarMultiplyForward));

PD_BUILD_GRAD_OP(custom_left_scalar_multiply)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(LeftScalarMultiplyBackward));

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

// y = 1 / x
std::vector<paddle::Tensor> LeftScalarDivideForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    return {1 / x};
  } else {
    PD_THROW("Not implemented.");
  }
}

// dy / dx = -grad_out / (x * x)
std::vector<paddle::Tensor> LeftScalarDivideBackward(
    const paddle::Tensor& x,
    const paddle::Tensor& out,
    const paddle::Tensor& grad_out) {
  if (x.is_cpu() || x.is_gpu()) {
    return {-grad_out / (x * x)};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_left_scalar_divide)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(LeftScalarDivideForward));

PD_BUILD_GRAD_OP(custom_left_scalar_divide)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(LeftScalarDivideBackward));

// out = x & y
std::vector<paddle::Tensor> AndForward(const paddle::Tensor& x,
                                       const paddle::Tensor& y) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x & y};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_logical_and)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AndForward));

// out = x | y
std::vector<paddle::Tensor> OrForward(const paddle::Tensor& x,
                                      const paddle::Tensor& y) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x | y};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_logical_or)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(OrForward));

// out = x ^ y
std::vector<paddle::Tensor> XorForward(const paddle::Tensor& x,
                                       const paddle::Tensor& y) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x ^ y};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_logical_xor)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(XorForward));

// out = ~x
std::vector<paddle::Tensor> NotForward(const paddle::Tensor& x) {
  if (x.is_cpu() || x.is_gpu()) {
    return {~x};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_logical_not)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(NotForward));

// out = (x < y)
std::vector<paddle::Tensor> LessThanForward(const paddle::Tensor& x,
                                            const paddle::Tensor& y) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x < y};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_less_than)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(LessThanForward));

// out = (x <= y)
std::vector<paddle::Tensor> LessEqualForward(const paddle::Tensor& x,
                                             const paddle::Tensor& y) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x <= y};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_less_equal)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(LessEqualForward));

// out = (x == y)
std::vector<paddle::Tensor> EqualForward(const paddle::Tensor& x,
                                         const paddle::Tensor& y) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x == y};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_equal)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(EqualForward));

// out = (x != y)
std::vector<paddle::Tensor> NotEqualForward(const paddle::Tensor& x,
                                            const paddle::Tensor& y) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x != y};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_not_equal)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(NotEqualForward));

// out = (x > y)
std::vector<paddle::Tensor> GreaterThanForward(const paddle::Tensor& x,
                                               const paddle::Tensor& y) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x > y};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_greater_than)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(GreaterThanForward));

// out = (x >= y)
std::vector<paddle::Tensor> GreaterEqualForward(const paddle::Tensor& x,
                                                const paddle::Tensor& y) {
  if (x.is_cpu() || x.is_gpu()) {
    return {x >= y};
  } else {
    PD_THROW("Not implemented.");
  }
}

PD_BUILD_OP(custom_greater_equal)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(GreaterEqualForward));
