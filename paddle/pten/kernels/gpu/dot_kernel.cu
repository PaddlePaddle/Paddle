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

#include "paddle/pten/kernels/dot_kernel.h"

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/funcs/eigen/common.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/platform/complex.h"

namespace pten {

template <typename T, typename Context>
void DotKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  out->mutable_data<T>();
  if (1 == out->dims().size()) {
    auto eigen_out = pten::EigenScalar<T>::From(*out);
    auto eigen_x = pten::EigenVector<T>::Flatten(x);
    auto eigen_y = pten::EigenVector<T>::Flatten(y);

    auto& dev = *dev_ctx.eigen_device();
    eigen_out.device(dev) = (eigen_x * eigen_y).sum();
  } else {
    auto eigen_out = pten::EigenMatrix<T>::From(*out);
    auto eigen_x = pten::EigenMatrix<T>::From(x);
    auto eigen_y = pten::EigenMatrix<T>::From(y);

    auto& dev = *dev_ctx.eigen_device();
    eigen_out.device(dev) = (eigen_x * eigen_y).sum(Eigen::DSizes<int, 1>(1));
  }
}

}  // namespace pten

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL(dot,
                   GPU,
                   ALL_LAYOUT,
                   pten::DotKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
