/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/reverse_kernel.h"

namespace phi {

template <typename Context, typename T, int Rank>
struct ReverseFunctor {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& in,
                  DenseTensor* out,
                  const IntArray& axis) {
    auto& axis_data = axis.GetData();
    Eigen::DSizes<bool, Rank> reverse_axis;
    for (int i = 0; i < Rank; ++i) {
      reverse_axis[i] = false;
    }
    for (int a : axis_data) {
      if (a >= 0) {
        reverse_axis[a] = true;
      } else {
        reverse_axis[Rank + a] = true;
      }
    }

    auto in_eigen = EigenTensor<T, Rank>::From(in);
    auto out_eigen = EigenTensor<T, Rank>::From(*out);
    auto& dev = *dev_ctx.eigen_device();

    funcs::EigenReverse<std::decay_t<decltype(dev)>, T, Rank>::Eval(
        dev, out_eigen, in_eigen, reverse_axis);
  }
};

template <typename T, typename Context>
void ReverseKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& axis,
                   DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  int rank = x.dims().size();

  switch (rank) {
    case 1:
      ReverseFunctor<Context, T, 1> functor1;
      functor1(dev_ctx, x, out, axis);
      break;
    case 2:
      ReverseFunctor<Context, T, 2> functor2;
      functor2(dev_ctx, x, out, axis);
      break;
    case 3:
      ReverseFunctor<Context, T, 3> functor3;
      functor3(dev_ctx, x, out, axis);
      break;
    case 4:
      ReverseFunctor<Context, T, 4> functor4;
      functor4(dev_ctx, x, out, axis);
      break;
    case 5:
      ReverseFunctor<Context, T, 5> functor5;
      functor5(dev_ctx, x, out, axis);
      break;
    case 6:
      ReverseFunctor<Context, T, 6> functor6;
      functor6(dev_ctx, x, out, axis);
      break;
    default:
      PADDLE_THROW(phi::errors::OutOfRange(
          "The reserve operator does not support input tensors"
          "whose ranks are greater than 6."));
  }
}

}  // namespace phi
