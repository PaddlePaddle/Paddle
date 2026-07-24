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
  out->Resize(shape.GetData());
  T *data = dev_ctx.template Alloc<T>(out);
  auto size = out->numel();

  // Bit-for-bit aligned with PyTorch's CPU `Tensor.uniform_`: with the same
  // seed, torch.Generator().manual_seed(seed) on the torch side produces an
  // identical bit pattern.
  funcs::TorchMT19937Engine engine(
      seed ? static_cast<uint64_t>(seed)
           : dev_ctx.GetGenerator()->GetCurrentSeed());

  if constexpr (std::is_same_v<T, dtype::complex<float>> ||
                std::is_same_v<T, dtype::complex<double>>) {
    // torch fills a complex tensor through view_as_real: an interleaved
    // re/im sequence of 2 * numel real samples drawn from the same stream.
    using RealType = dtype::Real<T>;  // float or double
    funcs::UniformRealDistributionTorchAligned<RealType>(
        reinterpret_cast<RealType *>(data),
        size * 2,
        min.to<double>(),
        max.to<double>(),
        &engine);
  } else {
    funcs::UniformRealDistributionTorchAligned<T>(
        data, size, min.to<double>(), max.to<double>(), &engine);
  }

  if (seed == 0 && size > 0) {
    // Advance the global RNG state so that consecutive calls produce
    // different sequences (same convention as cpu/randperm_kernel.cc).
    // Skipped for 0-size outputs: torch draws nothing from the generator
    // there, so the global state must stay untouched.
    dev_ctx.GetGenerator()->SetCurrentSeed(engine());
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
