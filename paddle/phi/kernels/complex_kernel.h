/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void ConjKernel(const Context& dev_ctx, const DenseTensor& x, DenseTensor* out);

template <typename T, typename Context>
void RealKernel(const Context& dev_ctx, const DenseTensor& x, DenseTensor* out);

template <typename T, typename Context>
void ImagKernel(const Context& dev_ctx, const DenseTensor& x, DenseTensor* out);

// If T is complex
template <
    typename T,
    typename Context,
    std::enable_if_t<std::is_same<T, phi::dtype::complex<float>>::value ||
                         std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
DenseTensor Conj(const Context& dev_ctx, const DenseTensor& x) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  UnchangedInferMeta(x, &meta_out);
  ConjKernel<T>(dev_ctx, x, &dense_out);
  return dense_out;
}

// If T is not complex
template <
    typename T,
    typename Context,
    std::enable_if_t<!std::is_same<T, phi::dtype::complex<float>>::value &&
                         !std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
DenseTensor Conj(const Context& dev_ctx, const DenseTensor& x) {
  return x;
}

// If T is complex
template <
    typename T,
    typename Context,
    std::enable_if_t<std::is_same<T, phi::dtype::complex<float>>::value ||
                         std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
DenseTensor Real(const Context& dev_ctx, const DenseTensor& x) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  RealAndImagInferMeta(x, &meta_out);
  RealKernel<T>(dev_ctx, x, &dense_out);
  return dense_out;
}

// If T is not complex
template <
    typename T,
    typename Context,
    std::enable_if_t<!std::is_same<T, phi::dtype::complex<float>>::value &&
                         !std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
DenseTensor Real(const Context& dev_ctx, const DenseTensor& x) {
  return x;
}

// If T is complex
template <
    typename T,
    typename Context,
    std::enable_if_t<std::is_same<T, phi::dtype::complex<float>>::value ||
                         std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
DenseTensor Imag(const Context& dev_ctx, const DenseTensor& x) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  RealAndImagInferMeta(x, &meta_out);
  ImagKernel<T>(dev_ctx, x, &dense_out);
  return dense_out;
}

// If T is not complex
template <
    typename T,
    typename Context,
    std::enable_if_t<!std::is_same<T, phi::dtype::complex<float>>::value &&
                         !std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
DenseTensor Imag(const Context& dev_ctx, const DenseTensor& x) {
  return x;
}

}  // namespace phi
