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

#include "paddle/phi/kernels/log_softmax_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrixTemplate = EigenMatrix<T, MajorType, IndexType>;

template <typename Context, typename T>
struct LogSoftmaxGradFunctor {
  void operator()(const Context& context,
                  const DenseTensor* Y,
                  const DenseTensor* dY,
                  DenseTensor* dX,
                  const int axis) {
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;

    const int n = funcs::SizeToAxis(axis, Y->dims());
    const int d = funcs::SizeFromAxis(axis, Y->dims());
    phi::DDim dim_2d{n, d};

    auto y = EigenMatrixTemplate<T>::From(*Y, dim_2d);
    auto dy = EigenMatrixTemplate<T>::From(*dY, dim_2d);
    auto dx = EigenMatrixTemplate<T>::From(*dX, dim_2d);

    const int axis_dim = Y->dims()[axis];
    const int batch_size = y.dimension(kBatchDim);
    const int num_classes = y.dimension(kClassDim);
    const int num_remain = num_classes / axis_dim;

    Eigen::DSizes<int, 1> along_class(kClassDim);
    Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);
    Eigen::DSizes<int, 2> one_axis(1, axis_dim);

    dx.device(*context.eigen_device()) =
        dy -
        (y.exp()) * (dy.reshape(batch_axis_remain)
                         .sum(along_class)
                         .broadcast(one_axis));
  }
};

template <typename T, typename Context>
void LogSoftmaxGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& out_grad,
                          int axis,
                          DenseTensor* x_grad) {
  const int rank = out.dims().size();
  const int canonical_axis = funcs::CanonicalAxis(axis, rank);

  dev_ctx.template Alloc<T>(x_grad);
  if (out.numel() != 0) {
    LogSoftmaxGradFunctor<Context, T>()(
        dev_ctx, &out, &out_grad, x_grad, canonical_axis);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(log_softmax_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::LogSoftmaxGradKernel,
                   float,
                   double) {}
