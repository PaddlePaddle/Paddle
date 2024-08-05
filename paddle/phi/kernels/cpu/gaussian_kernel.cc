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

#include "paddle/phi/kernels/gaussian_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/norm_distribution.h"

namespace phi {

template <typename T, typename Context>
void GaussianKernel(const Context& dev_ctx,
                    const IntArray& shape,
                    float mean,
                    float std,
                    int seed,
                    DataType dtype,
                    DenseTensor* out) {
  out->Resize(common::make_ddim(shape.GetData()));
  int64_t size = out->numel();
  T* data = dev_ctx.template Alloc<T>(out);
  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }
  NormalDistribution<T>(data, size, mean, std, engine);
}

template <typename T, typename Context>
void GaussianInplaceKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           float mean,
                           float std,
                           int seed,
                           DenseTensor* out) {
  T* data = dev_ctx.template Alloc<T>(out);

  int64_t size = out->numel();
  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }

  NormalDistribution<T>(data, size, mean, std, engine);
}

}  // namespace phi

PD_REGISTER_KERNEL(gaussian,
                   CPU,
                   ALL_LAYOUT,
                   phi::GaussianKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(gaussian_inplace,
                   CPU,
                   ALL_LAYOUT,
                   phi::GaussianInplaceKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
