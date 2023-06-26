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

#pragma once

#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/elementwise_grad_base.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T, typename Context>
void RealGradKernel(const Context& dev_ctx,
                    const DenseTensor& dout,
                    DenseTensor* dx) {
  auto numel = dout.numel();
  auto* dout_data = dout.data<phi::dtype::Real<T>>();
  auto* dx_data =
      dev_ctx.template Alloc<T>(dx, static_cast<size_t>(numel * sizeof(T)));

  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  phi::funcs::RealToComplexFunctor<T> functor(dout_data, dx_data, numel);
  for_range(functor);
}

template <typename T, typename Context>
void ImagGradKernel(const Context& dev_ctx,
                    const DenseTensor& dout,
                    DenseTensor* dx) {
  auto numel = dout.numel();
  auto* dout_data = dout.data<phi::dtype::Real<T>>();
  auto* dx_data =
      dev_ctx.template Alloc<T>(dx, static_cast<size_t>(numel * sizeof(T)));

  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  phi::funcs::ImagToComplexFunctor<T> functor(dout_data, dx_data, numel);
  for_range(functor);
}

template <typename T>
struct ComplexGradForRealFunctor {
  inline HOSTDEVICE T operator()(const T x UNUSED,
                                 const T y UNUSED,
                                 const phi::dtype::complex<T> out UNUSED,
                                 const phi::dtype::complex<T> dout) {
    return dout.real;
  }
};

template <typename T>
struct ComplexGradForImagFunctor {
  inline HOSTDEVICE T operator()(const T x UNUSED,
                                 const T y UNUSED,
                                 const phi::dtype::complex<T> out UNUSED,
                                 const phi::dtype::complex<T> dout) {
    return dout.imag;
  }
};

template <typename T, typename Context>
void ComplexGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       DenseTensor* dx,
                       DenseTensor* dy) {
  using C = phi::dtype::complex<T>;

  // skip out in a hacky way
  auto out = dout;
  phi::funcs::ElemwiseGradCompute<Context,
                                  T,
                                  ComplexGradForRealFunctor<T>,
                                  ComplexGradForImagFunctor<T>,
                                  C>(dev_ctx,
                                     x,
                                     y,
                                     out,
                                     dout,
                                     /*axis*/ -1,
                                     dx,
                                     dy,
                                     ComplexGradForRealFunctor<T>(),
                                     ComplexGradForImagFunctor<T>());
}

}  // namespace phi
