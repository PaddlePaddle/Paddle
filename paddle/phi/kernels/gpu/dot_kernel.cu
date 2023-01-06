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

#include "paddle/phi/kernels/dot_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/phi/common/complex.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T, typename Context>
void DotKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (1 == out->dims().size()) {
    auto eigen_out = phi::EigenScalar<T>::From(*out);
    auto eigen_x = phi::EigenVector<T>::Flatten(x);
    auto eigen_y = phi::EigenVector<T>::Flatten(y);

    auto& dev = *dev_ctx.eigen_device();
    eigen_out.device(dev) = (eigen_x * eigen_y).sum();
  } else {
    auto eigen_out = phi::EigenMatrix<T>::From(*out);
    auto eigen_x = phi::EigenMatrix<T>::From(x);
    auto eigen_y = phi::EigenMatrix<T>::From(y);

    auto& dev = *dev_ctx.eigen_device();
    eigen_out.device(dev) = (eigen_x * eigen_y).sum(Eigen::DSizes<int, 1>(1));
  }
}

}  // namespace phi

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(dot,
                   GPU,
                   ALL_LAYOUT,
                   phi::DotKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
