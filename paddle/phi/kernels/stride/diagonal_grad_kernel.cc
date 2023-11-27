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

#include "paddle/phi/kernels/diagonal_grad_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/diagonal_kernel.h"
#include "paddle/phi/kernels/fill_kernel.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"

namespace phi {

template <typename Context>
void DiagonalGradStridedKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& out_grad,
                               int offset,
                               int axis1,
                               int axis2,
                               DenseTensor* in_grad) {
  dev_ctx.Alloc(in_grad, in_grad->dtype());
  in_grad->set_strides(DenseTensorMeta::calc_strides(in_grad->dims()));
  PD_VISIT_ALL_TYPES(in_grad->dtype(), "DiagonalGradStridedKernel", ([&] {
                       phi::FillKernel<data_t, Context>(
                           dev_ctx, *in_grad, 0, in_grad);
                     }));
  DenseTensor tmp;
  tmp.set_layout(out_grad.layout());
  tmp.set_lod(out_grad.lod());
  tmp.set_type(out_grad.dtype());
  tmp.Resize(out_grad.dims());

  DiagonalStridedKernel<Context>(dev_ctx, *in_grad, offset, axis1, axis2, &tmp);
  PD_VISIT_ALL_TYPES(out_grad.dtype(), "DiagonalGradStridedKernel", ([&] {
                       phi::StridedCopyKernel<data_t, Context>(
                           dev_ctx,
                           out_grad,
                           phi::vectorize<int64_t>(tmp.dims()),
                           phi::vectorize<int64_t>(tmp.strides()),
                           tmp.offset(),
                           &tmp);
                     }));
}

}  // namespace phi
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    diagonal_grad, STRIDED, phi::DiagonalGradStridedKernel) {}
