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

#include "paddle/phi/kernels/index_select_grad_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fill_kernel.h"
#include "paddle/phi/kernels/index_select_kernel.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"
namespace phi {

template <typename Context>
void IndexSelectGradStridedKernel(const Context& dev_ctx,
                                  const DenseTensor& x,
                                  const DenseTensor& out_grad,
                                  int64_t index,
                                  int dim,
                                  DenseTensor* x_grad) {
  dev_ctx.Alloc(x_grad, x_grad->dtype());
  PD_VISIT_ALL_TYPES(x_grad->dtype(), "IndexSelectGradStridedKernel", ([&] {
                       phi::FillKernel<data_t, Context>(
                           dev_ctx, *x_grad, 0, x_grad);
                     }));
  DenseTensor tmp;
  IndexSelectStridedKernel<Context>(dev_ctx, *x_grad, index, dim, &tmp);
  PD_VISIT_ALL_TYPES(out_grad.dtype(), "IndexSelectGradStridedKernel", ([&] {
                       phi::StridedCopyKernel<data_t, Context>(
                           dev_ctx,
                           out_grad,
                           phi::vectorize<int64_t>(tmp.dims()),
                           phi::vectorize<int64_t>(tmp.stride()),
                           tmp.offset(),
                           &tmp);
                     }));
  x_grad->set_stride(DenseTensorMeta::calc_stride(x_grad->dims()));
}

}  // namespace phi
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    index_select_grad_strided, STRIDED, phi::IndexSelectGradStridedKernel) {}
