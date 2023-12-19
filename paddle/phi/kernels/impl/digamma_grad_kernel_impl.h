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

#include <unsupported/Eigen/SpecialFunctions>

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T>
struct DigammaGradFunctor {
  DigammaGradFunctor(const T* dout, const T* x, T* output, int64_t numel)
      : dout_(dout), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
    const MPType mp_dout = static_cast<MPType>(dout_[idx]);
    const MPType mp_x = static_cast<MPType>(x_[idx]);
    output_[idx] =
        static_cast<T>(mp_dout * Eigen::numext::polygamma(MPType(1), mp_x));
  }

 private:
  const T* dout_;
  const T* x_;
  T* output_;
  int64_t numel_;
};

template <typename T, typename Context>
void DigammaGradKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& out_grad,
                       DenseTensor* x_grad) {
  ctx.template Alloc<T>(x_grad);

  auto* dout_data = out_grad.data<T>();
  auto* x_data = x.data<T>();
  auto* dx_data = x_grad->data<T>();
  auto numel = out_grad.numel();
  phi::funcs::ForRange<Context> for_range(ctx, numel);
  DigammaGradFunctor<T> functor(dout_data, x_data, dx_data, numel);
  for_range(functor);
}

}  // namespace phi
