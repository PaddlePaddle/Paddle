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

#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/clip_kernel.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#endif

namespace phi {

template <typename T>
class ClipFunctor {
 public:
  explicit ClipFunctor(const T min, const T max) : min_(min), max_(max) {}
  HOSTDEVICE T operator()(const T x) const {
    return x < min_ ? min_ : x > max_ ? max_ : x;
  }

 private:
  T min_;
  T max_;
};

template <typename T, typename Context>
void ClipKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const Scalar& min,
                const Scalar& max,
                DenseTensor* out) {
  auto max_ = max.to<T>();
  auto min_ = min.to<T>();

  PADDLE_ENFORCE_LE(
      min_,
      max_,
      errors::InvalidArgument("max should be greater than or equal to min. "
                              "But received min = %f, max = %f",
                              static_cast<float>(min_),
                              static_cast<float>(max_)));

  T* out_data = dev_ctx.template Alloc<T>(out);
  // const T* x_data = x->data<T>();
  // int64_t numel = x->numel();
  const T* x_data = x.data<T>();
  int64_t numel = x.numel();
  if (paddle::platform::is_gpu_place(dev_ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
    std::vector<const DenseTensor*> ins = {&x};
    std::vector<DenseTensor*> outs = {out};
    auto functor = ClipFunctor<T>(min_, max_);
    phi::funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
#endif
  } else {
    paddle::platform::Transform<Context> trans;
    trans(
        dev_ctx, x_data, x_data + numel, out_data, ClipFunctor<T>(min_, max_));
  }
}

}  // namespace phi
