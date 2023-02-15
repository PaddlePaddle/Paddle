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

#include "paddle/phi/kernels/exponential_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"

namespace phi {

template <typename T, typename Context>
void ExponentialKernel(const Context &dev_ctx,
                       const DenseTensor &x,
                       float lambda,
                       DenseTensor *out) {
  phi::funcs::uniform_distribution<T> dist;
  phi::funcs::exponential_transform<T> trans(lambda);
  phi::funcs::distribution_and_transform<T>(dev_ctx, out, dist, trans);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    exponential, GPU, ALL_LAYOUT, phi::ExponentialKernel, float, double) {}
