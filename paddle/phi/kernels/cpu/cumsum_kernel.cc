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

#include "paddle/phi/kernels/cumsum_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

struct CumsumFunctor {
  template <typename X>
  const typename X::TensorScanSumOp operator()(X x,
                                               int axis,
                                               bool exclusive) const {
    return x.cumsum(axis, exclusive);
  }
};

template <typename Device, typename Dim, typename X, typename Out>
void ComputeImp(Device d,
                const Dim& dims,
                X x,
                Out out,
                int axis,
                bool reverse,
                bool exclusive) {
  if (!reverse) {
    out.reshape(dims).device(d) =
        CumsumFunctor()(x.reshape(dims), axis, exclusive);
  } else {
    std::array<bool, Dim::count> rev;
    rev.fill(false);
    rev[axis] = reverse;
    out.reshape(dims).device(d) =
        CumsumFunctor()(x.reshape(dims).reverse(rev), axis, exclusive)
            .reverse(rev);
  }
}

template <typename T, typename Context>
void CumsumKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int axis,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  DenseTensor* out) {
  auto out_dims = out->dims();

  PADDLE_ENFORCE_EQ(
      axis < out_dims.size() && axis >= (0 - out_dims.size()),
      true,
      phi::errors::OutOfRange(
          "Attr(axis) is out of range, It's expected "
          "to be in range of [-%d, %d]. But received Attr(axis) = %d.",
          out_dims.size(),
          out_dims.size() - 1,
          axis));
  if (axis < 0) {
    axis += out_dims.size();
  }

  dev_ctx.template Alloc<T>(out);

  int pre = 1;
  int post = 1;
  int mid = out_dims[axis];
  for (int i = 0; i < axis; ++i) {
    pre *= out_dims[i];
  }
  for (int i = axis + 1; i < out_dims.size(); ++i) {
    post *= out_dims[i];
  }

  auto x0 = EigenVector<T>::Flatten(x);
  auto out0 = EigenVector<T>::Flatten(*out);
  auto& place = *dev_ctx.eigen_device();

  using IndexT = Eigen::DenseIndex;
  if (pre == 1) {
    if (post == 1) {
      ComputeImp(place,
                 Eigen::DSizes<IndexT, 1>(mid),
                 x0,
                 out0,
                 /* axis= */ 0,
                 reverse,
                 exclusive);
    } else {
      ComputeImp(place,
                 Eigen::DSizes<IndexT, 2>(mid, post),
                 x0,
                 out0,
                 /* axis= */ 0,
                 reverse,
                 exclusive);
    }
  } else {
    if (post == 1) {
      ComputeImp(place,
                 Eigen::DSizes<IndexT, 2>(pre, mid),
                 x0,
                 out0,
                 /* axis= */ 1,
                 reverse,
                 exclusive);
    } else {
      ComputeImp(place,
                 Eigen::DSizes<IndexT, 3>(pre, mid, post),
                 x0,
                 out0,
                 /* axis= */ 1,
                 reverse,
                 exclusive);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cumsum,
                   CPU,
                   ALL_LAYOUT,
                   phi::CumsumKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t) {}
