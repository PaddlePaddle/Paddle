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

#include "paddle/phi/kernels/transpose_kernel.h"

#include <vector>

#include "gflags/gflags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/transpose_function.cu.h"
#include "paddle/phi/kernels/impl/transpose_grad_kernel_impl.h"
DECLARE_string(throw_strided_error_op);

namespace phi {
template <typename T, typename Context>
void TransposeKernel(const Context& ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     DenseTensor* out) {
  DenseTensor& xx = const_cast<DenseTensor&>(x);
  out->can_not_uses = xx.can_not_uses;
  if (*out->canNotUse == false) {
    *out->canNotUse = *xx.canNotUse;
  }
  xx.can_not_uses->insert(xx.canNotUse);
  xx.can_not_uses->insert(out->canNotUse);
  VLOG(1) << "stride api call log: TransposeKernel";

  if (FLAGS_throw_strided_error_op == "TransposeKernel") {
    PADDLE_THROW(phi::errors::PermissionDenied("wanghuan"));
  }
  size_t x_rank = x.dims().size();
  std::vector<int> formated_axis = axis;
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      formated_axis[i] = axis[i] + x_rank;
    }
  }

  ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  if (formated_axis.size() == 0) {
    phi::Copy<Context>(ctx, x, ctx.GetPlace(), false, out);
    return;
  }
  phi::funcs::TransposeGPUKernelDriver<T>(ctx, x, formated_axis, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(transpose,
                   GPU,
                   ALL_LAYOUT,
                   phi::TransposeKernel,
                   bool,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
