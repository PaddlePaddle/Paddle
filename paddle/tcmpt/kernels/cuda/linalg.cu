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

#include "paddle/tcmpt/kernels/cuda/linalg.h"

#include "paddle/tcmpt/core/kernel_registry.h"
#include "paddle/tcmpt/kernels/common/eigen/dot.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/complex.h"

namespace pt {

template <typename T>
void Dot(const CUDAContext& dev_ctx,
         const DenseTensor& x,
         const DenseTensor& y,
         DenseTensor* out) {
  eigen::Dot<CUDAContext, T>(dev_ctx, x, y, out);
}

}  // namespace pt

PT_REGISTER_MODULE(LinalgCUDA);

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL("dot",
                   CUDA,
                   Any,
                   pt::Dot,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
