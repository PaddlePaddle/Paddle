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

#include "paddle/pten/kernels/cpu/linalg.h"

#include "paddle/pten/core/kernel_registry.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/complex.h"

#include "paddle/pten/kernels/hybird/math/matmul_func.h"

namespace pten {

template <typename T>
void Dot(const CPUContext& dev_ctx,
         const DenseTensor& x,
         const DenseTensor& y,
         DenseTensor* out) {
  auto const *x_ptr = x.data<T>(), *x_ptr_ = &x_ptr[0];
  auto const *y_ptr = y.data<T>(), *y_ptr_ = &y_ptr[0];
  auto* z = out->mutable_data<T>();

  // Loop over the total N elements of both operands while sum-reducing every
  // B pairs along the way where B is the dimension of the least ordered axis
  auto&& d = x.dims();
  auto const N = x.numel();
  auto const B = d[d.size() - 1];

  for (int j = 0; j < N / B; j++) {
    T ss = 0;
    for (int i = 0; i < B; i++) ss += (*x_ptr_++) * (*y_ptr_++);
    z[j] = ss;
  }
}

template <typename T>
void Matmul(const CPUContext& dev_ctx,
            const DenseTensor& x,
            const DenseTensor& y,
            bool transpose_x,
            bool transpose_y,
            DenseTensor* out) {
  PADDLE_ENFORCE_NE(paddle::framework::product(x.dims()),
                    0,
                    paddle::platform::errors::InvalidArgument(
                        "The Input(X) dims size must not be equal 0,"
                        " but reviced dims size is 0. "));
  PADDLE_ENFORCE_NE(paddle::framework::product(y.dims()),
                    0,
                    paddle::platform::errors::InvalidArgument(
                        "The Input(Y) dims size must not be equal 0,"
                        " but reviced dims size is 0. "));
  math::MatMulFunction<CPUContext, T>(
      dev_ctx, x, y, out, transpose_x, transpose_y);
}

}  // namespace pten

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL(dot,
                   CPU,
                   ANY,
                   pten::Dot,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}

PT_REGISTER_KERNEL(
    matmul_v2, CPU, ANY, pten::Matmul, float, double, complex64, complex128) {}
