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

#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/elementwise_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {
namespace sparse {

template <typename Functor,
          typename T,
          typename DeviceContext,
          typename OutType = T>
class TransformFunctor {
 public:
  TransformFunctor(const DenseTensor &x,
                   const DenseTensor &y,
                   DenseTensor *z,
                   const DeviceContext &ctx,
                   Functor func,
                   const bool is_xsize_larger = true)
      : x_(x.data<T>()),
        y_(y.data<T>()),
        z_(ctx.template Alloc<OutType>(z)),
        nx_(x.numel()),
        ctx_(ctx),
        func_(func),
        is_xsize_larger_(is_xsize_larger) {
    if (is_xsize_larger_ == false) {
      nx_ = y.numel();
    }
  }

  inline void Run() const {
    paddle::platform::Transform<DeviceContext> trans;
    trans(ctx_, x_, x_ + nx_, y_, z_, func_);
  }

  inline void RunRowWise(int n, int pre) const {
    paddle::platform::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_,
            x_,
            x_ + nx_,
            RowwiseTransformIterator<T, DeviceContext>(y_, n),
            z_,
            func_);
    } else {
      trans(ctx_,
            y_,
            y_ + nx_,
            RowwiseTransformIterator<T, DeviceContext>(x_, n),
            z_,
            func_);
    }
  }

  inline void RunMidWise(int n, int pre, int post) const {
    paddle::platform::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_,
            x_,
            x_ + nx_,
            MidWiseTransformIterator<T, DeviceContext>(y_, n, post),
            z_,
            func_);
    } else {
      trans(ctx_,
            y_,
            y_ + nx_,
            MidWiseTransformIterator<T, DeviceContext>(x_, n, post),
            z_,
            func_);
    }
  }

 private:
  const T *x_;
  const T *y_;
  OutType *z_;
  int64_t nx_;
  const DeviceContext &ctx_;
  Functor func_;
  bool is_xsize_larger_;
};

}  // namespace funcs
}  // namespace sparse
}  // namespace phi
