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

#include <thrust/random.h>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"

namespace phi {

template <typename T>
using ComplexType = phi::dtype::complex<T>;

template <typename T>
struct GaussianGenerator {
  T mean_, std_;
  unsigned int seed_;
  unsigned int offset_ = 0;

  __host__ __device__ GaussianGenerator(T mean, T std, int seed)
      : mean_(mean), std_(std), seed_(seed) {}

  __host__ __device__ GaussianGenerator(T mean, T std, int seed, int offset)
      : mean_(mean), std_(std), seed_(seed), offset_(offset) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    thrust::normal_distribution<MT> dist(static_cast<MT>(mean_),
                                         static_cast<MT>(std_));
    unsigned int new_n = n + offset_;
    rng.discard(new_n);
    MT out = dist(rng);
    return static_cast<T>(out);
  }
};

template <typename T>
struct GaussianGenerator<ComplexType<T>> {
  T mean_, std_;
  unsigned int seed_;
  unsigned int offset_ = 0;

  __host__ __device__ GaussianGenerator(T mean, T std, int seed)
      : mean_(mean), std_(std), seed_(seed) {}

  __host__ __device__ GaussianGenerator(T mean, T std, int seed, int offset)
      : mean_(mean), std_(std), seed_(seed), offset_(offset) {}

  __host__ __device__ ComplexType<T> operator()(const unsigned int n) const {
    thrust::minstd_rand rng_real;
    thrust::minstd_rand rng_img;
    rng_real.seed(seed_);
    rng_img.seed(seed_);
    thrust::normal_distribution<T> dist(mean_, std_);
    unsigned int new_n = n + offset_;
    rng_real.discard(new_n);
    rng_img.discard(new_n);
    T real = dist(rng_real);
    T imag = dist(rng_img);
    return ComplexType<T>(real, imag);
  }
};

// If T is not complex
template <
    typename T,
    typename Context,
    std::enable_if_t<!std::is_same<T, phi::dtype::complex<float>>::value &&
                         !std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
void GaussianRandom(const Context& dev_ctx,
                    const IntArray& shape,
                    float mean,
                    float std,
                    int seed,
                    DataType dtype,
                    DenseTensor* out) {
  out->Resize(common::make_ddim(shape.GetData()));
  dev_ctx.template Alloc<T>(out);
  if (seed == 0) {
    // use global Generator seed
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    funcs::normal_distribution<MT> dist;
    funcs::normal_transform<MT> trans(static_cast<MT>(mean),
                                      static_cast<MT>(std));
    funcs::distribution_and_transform<T>(dev_ctx, out, dist, trans);
  } else {
    // use OP seed
    auto func =
        GaussianGenerator<T>(static_cast<T>(mean), static_cast<T>(std), seed);
    IndexKernel<T, GaussianGenerator<T>>(dev_ctx, out, func);
  }
}

// If T is complex
template <
    typename T,
    typename Context,
    std::enable_if_t<std::is_same<T, phi::dtype::complex<float>>::value ||
                         std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
void GaussianRandom(const Context& dev_ctx,
                    const IntArray& shape,
                    float mean,
                    float std,
                    int seed,
                    DataType dtype,
                    DenseTensor* out) {
  out->Resize(common::make_ddim(shape.GetData()));
  dev_ctx.template Alloc<T>(out);
  float std_of_real_or_imag = std::sqrt(std::pow(std, 2) / 2);
  if (seed == 0) {
    // use global Generator seed
    DenseTensor* out_real = new DenseTensor();
    DenseTensor* out_imag = new DenseTensor();
    out_real->Resize(common::make_ddim(shape.GetData()));
    out_imag->Resize(common::make_ddim(shape.GetData()));
    dev_ctx.template Alloc<T>(out_real);
    dev_ctx.template Alloc<T>(out_imag);
    funcs::normal_distribution<phi::dtype::Real<T>> dist;
    funcs::normal_distribution<phi::dtype::Real<T>> dist_imag;
    funcs::normal_transform<phi::dtype::Real<T>> trans(mean,
                                                       std_of_real_or_imag);
    funcs::distribution_and_transform<phi::dtype::Real<T>>(
        dev_ctx, out_real, dist, trans);
    funcs::distribution_and_transform<phi::dtype::Real<T>>(
        dev_ctx, out_imag, dist_imag, trans);
    phi::ComplexKernel<phi::dtype::Real<T>>(dev_ctx, *out_real, *out_imag, out);
  } else {
    // use OP seed
    auto func = GaussianGenerator<T>(mean, std_of_real_or_imag, seed);
    IndexKernel<T, GaussianGenerator<T>>(dev_ctx, out, func);
  }
}

// If T is not complex
template <
    typename T,
    typename Context,
    std::enable_if_t<!std::is_same<T, phi::dtype::complex<float>>::value &&
                         !std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
void GaussianRandomInplace(const Context& dev_ctx,
                           const DenseTensor& x,
                           float mean,
                           float std,
                           int seed,
                           DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (seed == 0) {
    // use global Generator seed
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    funcs::normal_distribution<MT> dist;
    funcs::normal_transform<MT> trans(static_cast<MT>(mean),
                                      static_cast<MT>(std));
    funcs::distribution_and_transform<T>(dev_ctx, out, dist, trans);
  } else {
    // use OP seed
    auto func =
        GaussianGenerator<T>(static_cast<T>(mean), static_cast<T>(std), seed);
    IndexKernel<T, GaussianGenerator<T>>(dev_ctx, out, func);
  }
}

// If T is complex
template <
    typename T,
    typename Context,
    std::enable_if_t<std::is_same<T, phi::dtype::complex<float>>::value ||
                         std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
void GaussianRandomInplace(const Context& dev_ctx,
                           const DenseTensor& x,
                           float mean,
                           float std,
                           int seed,
                           DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  float std_of_real_or_imag = std::sqrt(std::pow(std, 2) / 2);
  if (seed == 0) {
    // use global Generator seed
    DenseTensor* out_real = new DenseTensor();
    DenseTensor* out_imag = new DenseTensor();
    out_real->Resize(x.dims());
    out_imag->Resize(x.dims());
    dev_ctx.template Alloc<T>(out_real);
    dev_ctx.template Alloc<T>(out_imag);
    funcs::normal_distribution<phi::dtype::Real<T>> dist;
    funcs::normal_distribution<phi::dtype::Real<T>> dist_imag;
    funcs::normal_transform<phi::dtype::Real<T>> trans(mean,
                                                       std_of_real_or_imag);
    funcs::distribution_and_transform<phi::dtype::Real<T>>(
        dev_ctx, out_real, dist, trans);
    funcs::distribution_and_transform<phi::dtype::Real<T>>(
        dev_ctx, out_imag, dist_imag, trans);
    phi::ComplexKernel<phi::dtype::Real<T>>(dev_ctx, *out_real, *out_imag, out);
  } else {
    // use OP seed
    auto func = GaussianGenerator<T>(mean, std_of_real_or_imag, seed);
    IndexKernel<T, GaussianGenerator<T>>(dev_ctx, out, func);
  }
}

template <typename T, typename Context>
void GaussianKernel(const Context& dev_ctx,
                    const IntArray& shape,
                    float mean,
                    float std,
                    int seed,
                    DataType dtype,
                    DenseTensor* out) {
  GaussianRandom<T>(dev_ctx, shape, mean, std, seed, dtype, out);
}

template <typename T, typename Context>
void GaussianInplaceKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           float mean,
                           float std,
                           int seed,
                           DenseTensor* out) {
  GaussianRandomInplace<T>(dev_ctx, x, mean, std, seed, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(gaussian,
                   GPU,
                   ALL_LAYOUT,
                   phi::GaussianKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(gaussian_inplace,
                   GPU,
                   ALL_LAYOUT,
                   phi::GaussianInplaceKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
