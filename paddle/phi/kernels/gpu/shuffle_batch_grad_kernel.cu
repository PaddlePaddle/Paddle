// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#ifndef _MSC_VER
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#endif

#include "paddle/common/errors.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/gpu/shuffle_batch_utils.h"
#include "paddle/phi/kernels/shuffle_batch_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void ShuffleBatchGradKernel(const Context& dev_ctx,
                            const DenseTensor& shuffleidx,
                            const DenseTensor& out_grad,
                            int startup_seed,
                            DenseTensor* x_grad) {
#ifdef _MSC_VER
  PADDLE_THROW(common::errors::Unimplemented(
      "GPU shuffle_batch_grad is not supported on Windows yet"));
#else
  const auto* out_grad_data = out_grad.data<T>();
  const auto* shuffleidx_data = shuffleidx.data<int64_t>();
  auto* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  auto x_embed_size = x_grad->dims()[x_grad->dims().size() - 1];
  ReorderFunctor<T, false> functor(
      out_grad_data, shuffleidx_data, x_grad_data, x_embed_size);
  // TODO(zengjinle): for small data, direct cudaMemcpy may be better
  phi::funcs::ForRange<phi::GPUContext> for_range(dev_ctx, x_grad->numel());
  for_range(functor);
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(shuffle_batch_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ShuffleBatchGradKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}
#endif
