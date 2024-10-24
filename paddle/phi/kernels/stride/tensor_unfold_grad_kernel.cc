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
#include "paddle/phi/kernels/tensor_unfold_grad_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/strided_utils.h"
#include "paddle/phi/kernels/tensor_unfold_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void TensorUnfoldGradKernel(const Context& dev_ctx,
                            const DenseTensor& input,
                            const DenseTensor& out_grad,
                            int64_t axis,
                            int64_t size,
                            int64_t step,
                            DenseTensor* input_grad) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  if (axis < 0) {
    axis += input.dims().size();
  }
  dev_ctx.Alloc(input_grad, input_grad->dtype());
  input_grad->set_strides(DenseTensorMeta::calc_strides(input_grad->dims()));
  if (out_grad.numel() < input.numel()) {
    PD_VISIT_ALL_TYPES(input_grad->dtype(), "TensorUnfoldGradKernel", ([&] {
                         phi::StridedTensorFill<data_t>(
                             *input_grad, 0, input_grad);
                       }));
  }
  DenseTensor tmp;
  tmp.set_layout(out_grad.layout());
  tmp.set_lod(out_grad.lod());
  tmp.set_type(out_grad.dtype());
  tmp.Resize(out_grad.dims());

  TensorUnfoldKernel<Context>(dev_ctx, *input_grad, axis, size, step, &tmp);
  PD_VISIT_ALL_TYPES(out_grad.dtype(), "TensorUnfoldGradKernel", ([&] {
                       phi::StridedTensorCopy<data_t>(
                           out_grad,
                           common::vectorize<int64_t>(tmp.dims()),
                           common::vectorize<int64_t>(tmp.strides()),
                           tmp.offset(),
                           &tmp);
                     }));
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(tensor_unfold_grad,
                                         STRIDED,
                                         phi::TensorUnfoldGradKernel) {}
