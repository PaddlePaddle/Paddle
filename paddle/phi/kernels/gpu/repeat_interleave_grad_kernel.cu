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

#include "paddle/phi/kernels/repeat_interleave_grad_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/impl/repeat_interleave_grad_kernel_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
namespace phi {
template <typename T, typename Context>
void RepeatInterleaveGradKernelV2(const Context& dev_ctx,
                                  const DenseTensor& x,
                                  const DenseTensor& out_grad,
                                  int repeats,
                                  int dim,
                                  DenseTensor* x_grad) {
  if (x_grad && x_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  auto input_dim = x_grad->dims();
  auto output_grad_dim = out_grad.dims();

  const int ndim = input_dim.size();
  dim = (dim < 0) ? ndim + dim : dim;

  std::vector<int64_t> reshape_shape = vectorize(input_dim);
  reshape_shape.insert(reshape_shape.begin() + dim + 1, repeats);

  DenseTensor out_grad_copy;
  out_grad_copy.set_meta(out_grad.meta());
  out_grad_copy.ShareBufferWith(out_grad, true);

  out_grad_copy.Resize(make_ddim(reshape_shape));

  SumKernel<T, Context>(dev_ctx,
                        out_grad_copy,
                        phi::IntArray({dim + 1}),
                        x_grad->dtype(),
                        false,
                        x_grad);
}
}  // namespace phi

PD_REGISTER_KERNEL(repeat_interleave_with_tensor_index_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveWithTensorIndexGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(repeat_interleave_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveGradKernelV2,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
