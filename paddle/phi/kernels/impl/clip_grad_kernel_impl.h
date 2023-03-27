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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/clip_kernel.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#endif

namespace phi {

template <typename T>
class ClipGradFunctor {
 public:
  explicit ClipGradFunctor(const T min, const T max) : min_(min), max_(max) {}
  HOSTDEVICE T operator()(const T x, const T y) const {
    return (y > min_ && y < max_) ? x : static_cast<T>(0);
  }

 private:
  T min_;
  T max_;
};

template <typename T, typename Context>
void ClipGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    const Scalar& min,
                    const Scalar& max,
                    DenseTensor* x_grad) {
  auto max_ = max.to<T>();
  auto min_ = min.to<T>();

#if defined(__NVCC__) || defined(__HIPCC__)
  std::vector<const DenseTensor*> ins = {&out_grad, &x};
  std::vector<DenseTensor*> outs = {x_grad};
  auto functor = ClipGradFunctor<T>(min_, max_);
  dev_ctx.template Alloc<T>(x_grad);
  phi::funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
#else
  int64_t numel = out_grad.numel();
  auto* d_x_data = dev_ctx.template Alloc<T>(x_grad);
  const T* d_out_data = out_grad.data<T>();
  const T* x_data = x.data<T>();
  phi::Transform<Context> trans;
  trans(dev_ctx,
        d_out_data,
        d_out_data + numel,
        x_data,
        d_x_data,
        ClipGradFunctor<T>(min_, max_));
#endif
}

}  // namespace phi
