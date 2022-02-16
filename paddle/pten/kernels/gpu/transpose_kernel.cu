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

#include <vector>
#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/transpose_kernel.h"

#include "paddle/fluid/framework/gpu_utils.h"
#include "paddle/fluid/operators/transpose_op.cu.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/backends/gpu/gpu_launch_config.h"
#include "paddle/pten/common/bfloat16.h"
#include "paddle/pten/kernels/impl/transpose_grad_kernel_impl.h"

namespace pten {
template <typename T, typename Context>
void TransposeKernel(const Context& ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     DenseTensor* out) {
  int rank = axis.size();
  ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  paddle::operators::TransposeGPUKernelDriver<T>(ctx, rank, x, axis, out);
}

}  // namespace pten

PT_REGISTER_KERNEL(transpose,
                   GPU,
                   ALL_LAYOUT,
                   pten::TransposeKernel,
                   bool,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   paddle::platform::float16,
                   pten::dtype::bfloat16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}
