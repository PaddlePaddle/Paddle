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

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/kernels/abs_grad_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

#if defined(__NVCC__)

template <typename T>
struct AbsGradCUDAFunctor {
  HOSTDEVICE inline AbsGradCUDAFunctor() {}

  HOSTDEVICE inline T operator()(const T x, const T dout) const {
    T output;
    if (x == T(0)) {
      output = T(0);
    } else {
      output = T(dout) * (x / T(std::abs(x)));
    }
    return output;
  }
};

template <>
struct AbsGradCUDAFunctor<phi::dtype::bfloat16> {
  HOSTDEVICE inline AbsGradCUDAFunctor() {}

  HOSTDEVICE inline phi::dtype::bfloat16 operator()(
      const phi::dtype::bfloat16 x, const phi::dtype::bfloat16 dout) const {
    phi::dtype::bfloat16 output;
    if (x == phi::dtype::bfloat16(0)) {
      output = static_cast<phi::dtype::bfloat16>(0);
    } else {
      output = (dout) * (x / abs(x));
    }
    return output;
  }
};

template <>
struct AbsGradCUDAFunctor<phi::dtype::complex<float>> {
  HOSTDEVICE inline AbsGradCUDAFunctor() {}
  HOSTDEVICE inline phi::dtype::complex<float> operator()(
      const phi::dtype::complex<float> x, const float dout) const {
    phi::dtype::complex<float> output;
    if (x == phi::dtype::complex<float>(0)) {
      output = phi::dtype::complex<float>(0);
    } else {
      output = phi::dtype::complex<float>(dout) *
               (x / phi::dtype::complex<float>(abs(x)));
    }
    return output;
  }
};

template <>
struct AbsGradCUDAFunctor<phi::dtype::complex<double>> {
  HOSTDEVICE inline AbsGradCUDAFunctor() {}
  HOSTDEVICE inline phi::dtype::complex<double> operator()(
      const phi::dtype::complex<double> x, const double dout) const {
    phi::dtype::complex<double> output;
    if (x == phi::dtype::complex<double>(0)) {
      output = phi::dtype::complex<double>(0);
    } else {
      output = phi::dtype::complex<double>(dout) *
               (x / phi::dtype::complex<double>(abs(x)));
    }
    return output;
  }
};

template <typename T>
void AbsGradKernelImpl(const GPUContext& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& dout,
                       DenseTensor* dx) {
  std::vector<const DenseTensor*> ins = {&x, &dout};
  std::vector<DenseTensor*> outs = {dx};
  dev_ctx.Alloc<T>(dx);
  AbsGradCUDAFunctor<T> abs_grad_cuda_functor;
  phi::funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, abs_grad_cuda_functor);
}

template <typename T, typename Context>
void AbsGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& dout,
                   DenseTensor* dx) {
  AbsGradKernelImpl<T>(dev_ctx, x, dout, dx);
}
#else
template <typename T, typename Context>
void AbsGradKernel(const Context& ctx,
                   const DenseTensor& x,
                   const DenseTensor& dout,
                   DenseTensor* dx) {
  auto numel = dout.numel();
  auto* dout_data = dout.data<phi::dtype::Real<T>>();
  auto* x_data = x.data<T>();

  ctx.template Alloc<T>(dx, static_cast<size_t>(numel * sizeof(T)));
  auto* dx_data = dx->data<T>();

  phi::funcs::ForRange<Context> for_range(ctx, numel);
  phi::funcs::AbsGradFunctor<T> functor(dout_data, x_data, dx_data, numel);
  for_range(functor);
}

#endif
template <typename T, typename Context>
void AbsDoubleGradKernel(const Context& ctx,
                         const DenseTensor& x,
                         const DenseTensor& ddx,
                         DenseTensor* ddout) {
  auto numel = ddx.numel();
  auto* ddx_data = ddx.data<T>();
  auto* x_data = x.data<T>();
  ctx.template Alloc<T>(ddout, static_cast<size_t>(numel * sizeof(T)));
  auto* ddout_data = ddout->data<T>();

  phi::funcs::ForRange<Context> for_range(ctx, numel);
  phi::funcs::AbsGradGradFunctor<T> functor(
      ddx_data, x_data, ddout_data, numel);
  for_range(functor);
}

}  // namespace phi
