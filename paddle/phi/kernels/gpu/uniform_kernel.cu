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

#include <thrust/random.h>
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/kernels/complex_kernel.h"

#include "paddle/common/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"

namespace phi {

template <typename T>
struct UniformGenerator {
  T min_, max_;
  unsigned int seed_;
  T diag_val_;
  unsigned int diag_num_;
  unsigned int diag_step_;
  __host__ __device__ UniformGenerator(
      T min, T max, int seed, int diag_num, int diag_step, T diag_val)
      : min_(min),
        max_(max),
        seed_(seed),
        diag_num_(diag_num),
        diag_step_(diag_step),
        diag_val_(diag_val) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(min_, max_);
    rng.discard(n);
    T out = dist(rng);
    unsigned int remainder = n % (diag_step_ + 1);
    if (remainder == 0 && diag_num_ > n / (diag_step_ + 1)) {
      out = diag_val_;
    }
    return out;
  }
};

template <typename T, typename Context>
void UniformKernel(const Context& dev_ctx,
                   const IntArray& shape,
                   DataType dtype,
                   const Scalar& min,
                   const Scalar& max,
                   int seed,
                   DenseTensor* out) {
  out->Resize(common::make_ddim(shape.GetData()));
  dev_ctx.template Alloc<T>(out);

  // Handle complex types separately
  const bool is_complex = std::is_same_v<T, phi::dtype::complex<float>> ||
                          std::is_same_v<T, phi::dtype::complex<double>>;

  if (is_complex) {
    using RealType = phi::dtype::Real<T>;  // float or double
    RealType min_val = min.to<RealType>();
    RealType max_val = max.to<RealType>();

    if (seed == 0) {
      // Use global Generator seed
      funcs::uniform_distribution<RealType> dist;
      funcs::uniform_real_transform<RealType> trans(min_val, max_val);

      // Generate random values for real and imaginary parts separately
      DenseTensor real_part, imag_part;
      real_part.Resize(out->dims());
      imag_part.Resize(out->dims());
      dev_ctx.template Alloc<RealType>(&real_part);
      dev_ctx.template Alloc<RealType>(&imag_part);

      funcs::distribution_and_transform<RealType>(
          dev_ctx, &real_part, dist, trans);
      funcs::distribution_and_transform<RealType>(
          dev_ctx, &imag_part, dist, trans);

      // Combine real and imaginary parts using ComplexKernel
      ComplexKernel<RealType, Context>(dev_ctx, real_part, imag_part, out);
    } else {
      // Use OP seed
      // Define the device lambda outside constexpr if to avoid MSVC restriction
      auto func = [=] __device__(int64_t idx) {
        thrust::minstd_rand engine;
        engine.seed(seed);
        engine.discard(idx * 2);
        thrust::uniform_real_distribution<RealType> dist(min_val, max_val);
        RealType real_val = dist(engine);
        RealType imag_val = dist(engine);
        return T(real_val, imag_val);
      };  // NOLINT(readability/braces)

      IndexKernel<T, decltype(func)>(dev_ctx, out, func);
    }
  } else {
    // Original implementation for non-complex types
    if (seed == 0) {
      // Use global Generator seed
      using MT = typename phi::dtype::MPTypeTrait<T>::Type;
      funcs::uniform_distribution<MT> dist;
      funcs::uniform_real_transform<MT> trans(min.to<float>(), max.to<float>());
      funcs::distribution_and_transform<T>(dev_ctx, out, dist, trans);
    } else {
      // Use OP seed
      auto func = UniformGenerator<T>(static_cast<T>(min.to<float>()),
                                      static_cast<T>(max.to<float>()),
                                      seed,
                                      0,
                                      0,
                                      static_cast<T>(0.0));
      IndexKernel<T, UniformGenerator<T>>(dev_ctx, out, func);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform,
                   GPU,
                   ALL_LAYOUT,
                   phi::UniformKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   phi::float8_e4m3fn,
                   phi::complex64,
                   phi::complex128) {}
