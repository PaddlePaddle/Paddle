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

#include "paddle/top/core/dense_tensor.h"
#include "paddle/top/core/kernel_registry.h"
#include "paddle/top/module/scale.h"
#include "paddle/top/module/sign.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/device_context.h"

namespace pt {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = paddle::framework::EigenScalar<T, MajorType, IndexType>;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = paddle::framework::EigenVector<T, MajorType, IndexType>;

using CPUContext = paddle::platform::CPUDeviceContext;

/**
 * [ How do we organize the kernel directory ]
 * Now according to the classification of operators in the Python API,
 * the same type of operation kernel is placed in a header file.
 * This is only a temporary approach.
 *
 * Considerations:
 *
 * 1. In the future, it may be tailored the lib on kernel level.
 *    This organization will cause difficulty in tailoring;
 * 2. If there is still one *.h and *.cc file for one kernel,
 *    and now the kernel is organized by device, the number of files
 *    will be greatly expanded, but this may be more reasonable;
 * 3. In the future, the kernel implementation of the function should
 *    be in the *.cc file. If you want to call the kernel in the tensor
 *    operation library, you should find the call through the global
 *    KernelMap instead of including the header file of the corresponding
 *    calculation. This may reduce the number of header files.
 */

template <typename T>
void Sign(const CPUContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  module::Sign<CPUContext, T>(dev_ctx, x, out);
}

template <typename T>
void Mean(const CPUContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  out->mutable_data<T>();
  auto x_data = EigenVector<T>::Flatten(x);
  auto y_data = EigenScalar<T>::From(*out);
  auto& place = *dev_ctx.eigen_device();
  y_data.device(place) = x_data.mean();
}

template <typename T>
void Scale(const CPUContext& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  module::Scale<CPUContext, T>(dev_ctx, x, scale, bias, bias_after_scale, out);
}

}  // namespace pt

PT_DECLARE_KERNEL_2T(sign, CPU, NCHW, float, double);
