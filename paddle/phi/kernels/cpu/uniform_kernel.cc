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

#include "paddle/phi/kernels/uniform_kernel.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/funcs/uniform_real_distribution.h"

namespace phi {

template <typename T, typename Context>
void UniformKernel(const Context &dev_ctx,
                   const IntArray &shape,
                   DataType dtype UNUSED,
                   const Scalar &min,
                   const Scalar &max,
                   int seed,
                   DenseTensor *out) {
  out->Resize(common::make_ddim(shape.GetData()));
  T *data = dev_ctx.template Alloc<T>(out);
  auto size = out->numel();

  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }

  // Handle complex types separately
  if constexpr (std::is_same_v<T, phi::dtype::complex<float>> ||
                std::is_same_v<T, phi::dtype::complex<double>>) {
    using RealType = phi::dtype::Real<T>;  // float or double
    RealType min_val = min.to<RealType>();
    RealType max_val = max.to<RealType>();

    // Generate random values for real and imaginary parts separately
    DenseTensor real_part, imag_part;
    real_part.Resize(out->dims());
    imag_part.Resize(out->dims());
    RealType *real_data = dev_ctx.template Alloc<RealType>(&real_part);
    RealType *imag_data = dev_ctx.template Alloc<RealType>(&imag_part);

    // Generate real part
    UniformRealDistribution<RealType>(
        real_data, size, min_val, max_val, engine);

    // Generate imaginary part
    UniformRealDistribution<RealType>(
        imag_data, size, min_val, max_val, engine);

    // Combine real and imaginary parts using ComplexKernel
    ComplexKernel<RealType, Context>(dev_ctx, real_part, imag_part, out);
  } else {
    // Original implementation for non-complex types
    UniformRealDistribution<T>(
        data, size, min.to<float>(), max.to<float>(), engine);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform,
                   CPU,
                   ALL_LAYOUT,
                   phi::UniformKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
