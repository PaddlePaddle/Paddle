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

#include "paddle/pten/kernels/cuda/linalg.h"

#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/hybird/eigen/dot.h"
#include "paddle/pten/kernels/hybird/math/matmul_func.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/complex.h"

namespace pten {

template <typename T>
void Dot(const CUDAContext& dev_ctx,
         const DenseTensor& x,
         const DenseTensor& y,
         DenseTensor* out) {
  eigen::Dot<CUDAContext, T>(dev_ctx, x, y, out);
}

template <typename T>
void Matmul(const CUDAContext& dev_ctx,
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
  math::MatMulFunction<CUDAContext, T>(
      dev_ctx, x, y, out, transpose_x, transpose_y);
}

}  // namespace pten

using float16 = paddle::platform::float16;
using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL(dot,
                   CUDA,
                   ANY,
                   pten::Dot,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}

PT_REGISTER_KERNEL(matmul_v2,
                   CUDA,
                   ANY,
                   pten::Matmul,
                   float,
                   double,
                   float16,
                   complex64,
                   complex128) {}
