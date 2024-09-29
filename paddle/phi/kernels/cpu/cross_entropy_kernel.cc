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

#include "paddle/phi/kernels/cross_entropy_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/cross_entropy.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/softmax_kernel.h"

namespace phi {

template <typename T>
void CrossEntropy(const CPUContext& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& label,
                  bool soft_label,
                  int ignore_index,
                  int axis,
                  DenseTensor* out) {
  const int rank = x.dims().size();
  const int axis_v = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = static_cast<int>(x.dims()[axis_v]);

  PADDLE_ENFORCE_GT(
      axis_dim,
      0,
      common::errors::InvalidArgument(
          "The axis dimension should be larger than 0, but received "
          "axis dimension is %d.",
          axis_dim));

  dev_ctx.template Alloc<T>(out);

  const int n = phi::funcs::SizeToAxis(axis_v, x.dims());
  PADDLE_ENFORCE_GT(
      n,
      0,
      common::errors::InvalidArgument(
          "The size of axis should be larger than 0, but received "
          "SizeToAxis of softmax is %d.",
          n));

  const int d = phi::funcs::SizeFromAxis(axis_v, x.dims());

  DenseTensor x_2d(x);
  x_2d.Resize({n, d});
  DenseTensor label_2d(label);
  label_2d.Resize({n, label.numel() / n});
  DenseTensor out_2d(*out);
  out_2d.Resize({n, d / axis_dim});

  phi::funcs::CrossEntropyFunctor<CPUContext, T>()(
      dev_ctx, &out_2d, &x_2d, &label_2d, soft_label, ignore_index, axis_dim);
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxKernel(const Context& dev_ctx,
                                   const DenseTensor& logits,
                                   const DenseTensor& label,
                                   bool soft_label,
                                   bool use_softmax,
                                   bool numeric_stable_mode,
                                   int ignore_index,
                                   int axis,
                                   DenseTensor* softmax,
                                   DenseTensor* loss) {
  // do not with softmax op, and input is softmax
  if (!use_softmax) {
    CrossEntropy<T>(
        dev_ctx, logits, label, soft_label, ignore_index, axis, loss);
    // cause of input is softmax, copy to output softmax, directly
    phi::Copy<Context>(dev_ctx, logits, dev_ctx.GetPlace(), false, softmax);
    return;
  }

  phi::SoftmaxKernel<T, Context>(dev_ctx, logits, axis, softmax);
  CrossEntropy<T>(
      dev_ctx, *softmax, label, soft_label, ignore_index, axis, loss);
}

}  // namespace phi

PD_REGISTER_KERNEL(cross_entropy_with_softmax,
                   CPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxKernel,
                   float,
                   double) {}
