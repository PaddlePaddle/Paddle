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

#include "paddle/phi/kernels/log_softmax_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrixTemplate = EigenMatrix<T, MajorType, IndexType>;

template <typename T>
struct ValueClip {
  HOSTDEVICE T operator()(const T& x) const {
    const T kThreshold = static_cast<T>(-64.);
    return x < kThreshold ? kThreshold : x;
  }
};

template <typename Context, typename T>
struct LogSoftmaxFunctor {
  void operator()(const Context& context,
                  const DenseTensor* X,
                  DenseTensor* Y,
                  const int axis) {
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;
    constexpr int kAxisDim = 1;

    int axis_dim = static_cast<int>(X->dims()[axis]);
    const int n = funcs::SizeToAxis(axis, X->dims());
    const int d = funcs::SizeFromAxis(axis, X->dims());
    phi::DDim dim_2d{n, d};

    auto logits = EigenMatrixTemplate<T>::From(*X, dim_2d);
    auto log_softmax = EigenMatrixTemplate<T>::From(*Y, dim_2d);

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);
    const int num_remain = num_classes / axis_dim;

    Eigen::DSizes<int, 1> along_axis(kAxisDim);
    Eigen::DSizes<int, 2> batch_classes(batch_size, num_classes);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);
    Eigen::DSizes<int, 3> batch_one_remain(batch_size, 1, num_remain);
    Eigen::DSizes<int, 3> one_axis_one(1, axis_dim, 1);
    Eigen::DSizes<int, 2> one_axis(1, axis_dim);
    Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);

    // For numerical stability, logits should be shifted by maximum number along
    // axis, calculate shifted_logits into log_softmax tensor for memory reuse.
    if (num_remain == 1) {
      // axis == -1, axis and class in same dimension, calculate along
      // class dimension directly for higher performance
      log_softmax.device(*context.eigen_device()) =
          (logits - logits.maximum(along_axis)
                        .eval()
                        .reshape(batch_by_one)
                        .broadcast(one_by_class))
              .unaryExpr(ValueClip<T>());
    } else {
      // axis != -1, class dimension split into (axis, remain), max and sum
      // should be calculated along axis dimension
      log_softmax.device(*context.eigen_device()) =
          (logits.reshape(batch_axis_remain) - logits.reshape(batch_axis_remain)
                                                   .maximum(along_axis)
                                                   .eval()
                                                   .reshape(batch_one_remain)
                                                   .broadcast(one_axis_one)
                                                   .reshape(batch_classes))
              .unaryExpr(ValueClip<T>());
    }

    log_softmax.device(*context.eigen_device()) =
        log_softmax - log_softmax.exp()
                          .eval()
                          .reshape(batch_axis_remain)
                          .sum(along_axis)
                          .log()
                          .broadcast(one_axis);
  }
};

template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out) {
  const int rank = x.dims().size();
  const int canonical_axis = funcs::CanonicalAxis(axis, rank);

  dev_ctx.template Alloc<T>(out);
  // For 0D Tensor
  if (rank == 0) {
    phi::funcs::set_constant(dev_ctx, out, static_cast<T>(0.0));
    return;
  }
  if (x.numel() != 0) {
    LogSoftmaxFunctor<Context, T>()(dev_ctx, &x, out, canonical_axis);
  }
}

}  // namespace phi

// TODO(YuanRisheng): The layout of onednn kernel should be OneDNN, we should
// support specifying the exact layout when the kernel is registered
PD_REGISTER_KERNEL(
    log_softmax, CPU, ALL_LAYOUT, phi::LogSoftmaxKernel, float, double) {}
