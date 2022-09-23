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

#include "paddle/phi/kernels/mean_all_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void MeanAllKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto X = EigenVector<T>::Flatten(x);
  auto y = EigenScalar<T>::From(*out);
  auto& place = *dev_ctx.eigen_device();

  y.device(place) = X.mean();
}

}  // namespace phi

PD_REGISTER_KERNEL(mean_all,
                   CPU,
                   ALL_LAYOUT,
                   phi::MeanAllKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
