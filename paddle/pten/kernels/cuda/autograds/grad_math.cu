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

#include "paddle/pten/kernels/cuda/autograds/grad_math.h"

#include "paddle/pten/core/kernel_registry.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/eigen.h"
// #include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/pten/kernels/functions/autograds/grad_elementwise_function.h"

namespace pten {

template <typename T>
void GradElementwiseAdd(const CUDAContext& dev_ctx,
                        const DenseTensor& X,
                        const DenseTensor& Y,
                        const DenseTensor& GradOut,
                        int axis,
                        DenseTensor* GradX,
                        DenseTensor* GradY) {
  math::ElementwiseAddGradFunction<CUDAContext, T>(
      dev_ctx, X, Y, GradOut, axis, GradX, GradY);
}

}  // namespace pten

PT_REGISTER_MODULE(GradElementwiseCUDA);

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL("grad_elementwise_add",
                   CUDA,
                   ANY,
                   pten::GradElementwiseAdd,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
