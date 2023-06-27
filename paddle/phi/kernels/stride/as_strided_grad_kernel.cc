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

#include "paddle/phi/kernels/as_strided_grad_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/as_strided_kernel.h"
#include "paddle/phi/kernels/fill_kernel.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"

namespace phi {

template <typename Context>
void AsStridedGradKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& out_grad,
                         const std::vector<int64_t>& dims,
                         const std::vector<int64_t>& stride,
                         int64_t offset,
                         DenseTensor* input_grad) {
  dev_ctx.Alloc(input_grad, input_grad->dtype());
  input_grad->set_strides(DenseTensorMeta::calc_strides(input_grad->dims()));
  PD_VISIT_ALL_TYPES(input_grad->dtype(), "AsStridedGradKernel", ([&] {
                       phi::FillKernel<data_t, Context>(
                           dev_ctx, *input_grad, 0, input_grad);
                     }));
  DenseTensor tmp;
  tmp.set_meta(out_grad.meta());
  AsStridedKernel<Context>(dev_ctx, *input_grad, dims, stride, offset, &tmp);
  PD_VISIT_ALL_TYPES(out_grad.dtype(), "AsStridedGradKernel", ([&] {
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
    as_strided_grad, STRIDED, phi::AsStridedGradKernel) {}
