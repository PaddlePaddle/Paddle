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

#pragma once

// See Note [ Why still include the fluid headers? ]
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T, typename Context>
void ConjKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DenseTensor* out) {
  auto numel = x.numel();
  if (x.numel() == 0) {
    auto x_dims = x.dims();
    out->Resize(x_dims);
    dev_ctx.template Alloc<T>(out);
    return;
  }
  auto* x_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);

  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  phi::funcs::ConjFunctor<T> functor(x_data, numel, out_data);
  for_range(functor);
}

template <typename T, typename Context>
void RealKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DenseTensor* out) {
  auto numel = x.numel();
  auto* x_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<phi::dtype::Real<T>>(
      out, static_cast<size_t>(numel * sizeof(phi::dtype::Real<T>)));

  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  phi::funcs::RealFunctor<T> functor(x_data, out_data, numel);
  for_range(functor);
}

template <typename T, typename Context>
void ImagKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DenseTensor* out) {
  auto numel = x.numel();
  auto* x_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<phi::dtype::Real<T>>(
      out, static_cast<size_t>(numel * sizeof(phi::dtype::Real<T>)));

  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  phi::funcs::ImagFunctor<T> functor(x_data, out_data, numel);
  for_range(functor);
}

// functors to use with ElementwiseComputeEx
template <typename T>
struct RealAndImagToComplexFunctor {
  inline HOSTDEVICE phi::dtype::complex<T> operator()(const T x, const T y) {
    return phi::dtype::complex<T>(x, y);
  }
};

template <typename T>
struct ImagAndRealToComplexFunctor {
  inline HOSTDEVICE phi::dtype::complex<T> operator()(const T y, const T x) {
    return phi::dtype::complex<T>(x, y);
  }
};

template <typename T, typename Context>
void ComplexKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  using C = phi::dtype::complex<T>;
  dev_ctx.template Alloc<C>(out);

// NOTE(chenfeiyu): be careful of the caveats of calling elementwise-related
// facility functions
#if defined(__NVCC__) || defined(__HIPCC__)
  phi::funcs::ElementwiseCompute<RealAndImagToComplexFunctor<T>, T, C>(
      dev_ctx, x, y, RealAndImagToComplexFunctor<T>(), out);
#else
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  if (x_dims.size() >= y_dims.size()) {
    phi::funcs::ElementwiseCompute<RealAndImagToComplexFunctor<T>, T, C>(
        dev_ctx, x, y, RealAndImagToComplexFunctor<T>(), out);
  } else {
    phi::funcs::ElementwiseCompute<ImagAndRealToComplexFunctor<T>, T, C>(
        dev_ctx, x, y, ImagAndRealToComplexFunctor<T>(), out);
  }
#endif
}

}  // namespace phi
