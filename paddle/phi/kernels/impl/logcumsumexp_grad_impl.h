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

#include <limits>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cum_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
struct LogGradPositiveFunctor {
  HOSTDEVICE T operator()(const T& x) const {
    const T kMin = std::numeric_limits<T>::lowest();
    return x > 0 ? std::log(x) : kMin;
  }
};

template <typename T>
struct LogGradNegativeFunctor {
  HOSTDEVICE T operator()(const T& x) const {
    const T kMin = std::numeric_limits<T>::lowest();
    return x < 0 ? std::log(-x) : kMin;
  }
};

template <typename T, typename Context>
void LogcumsumexpGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& out,
                            const DenseTensor& d_out,
                            int axis,
                            bool flatten,
                            bool exclusive,
                            bool reverse,
                            DenseTensor* d_x) {
  reverse = !reverse;
  dev_ctx.template Alloc<T>(d_x);

  auto eigen_x = EigenVector<T>::Flatten(x);
  auto eigen_out = EigenVector<T>::Flatten(out);
  auto eigen_d_out = EigenVector<T>::Flatten(d_out);
  auto& place = *dev_ctx.eigen_device();

  DenseTensor output_pos;
  output_pos.Resize(d_out.dims());
  dev_ctx.template Alloc<T>(&output_pos);
  auto eigen_output_pos = EigenVector<T>::Flatten(output_pos);
  DenseTensor output_neg;
  output_neg.Resize(d_out.dims());
  dev_ctx.template Alloc<T>(&output_neg);
  auto eigen_output_neg = EigenVector<T>::Flatten(output_neg);
  DenseTensor tmp;
  tmp.Resize(d_out.dims());
  dev_ctx.template Alloc<T>(&tmp);
  auto eigen_tmp = EigenVector<T>::Flatten(tmp);

  eigen_tmp.device(place) =
      eigen_d_out.unaryExpr(LogGradPositiveFunctor<T>()) - eigen_out;
  LogcumsumexpKernel<T, Context>(
      dev_ctx, tmp, axis, flatten, exclusive, reverse, &output_pos);
  eigen_output_pos.device(place) = (eigen_output_pos + eigen_x).exp();

  eigen_tmp.device(place) =
      eigen_d_out.unaryExpr(LogGradNegativeFunctor<T>()) - eigen_out;
  LogcumsumexpKernel<T, Context>(
      dev_ctx, tmp, axis, flatten, exclusive, reverse, &output_neg);
  eigen_output_neg.device(place) = (eigen_output_neg + eigen_x).exp();

  auto eigen_d_x = EigenVector<T>::Flatten(*d_x);
  eigen_d_x.device(place) = eigen_output_pos - eigen_output_neg;
}
}  // namespace phi
