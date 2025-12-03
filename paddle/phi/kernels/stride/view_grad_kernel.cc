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
#include "paddle/phi/kernels/view_grad_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/strided_reshape_utils.h"
#include "paddle/phi/kernels/funcs/strided_utils.h"
#include "paddle/phi/kernels/view_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void ViewShapeGradKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& out_grad,
                         const std::vector<int64_t>& dims,
                         DenseTensor* input_grad) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  DDim target_dims = input.dims();
  DDim target_stride;

  if (ReshapeStride(
          out_grad.dims(), out_grad.strides(), target_dims, target_stride)) {
    input_grad->set_meta(out_grad.meta());
    input_grad->Resize(target_dims);
    input_grad->set_strides(target_stride);
    input_grad->set_offset(out_grad.offset());
    input_grad->ResetHolder(out_grad.Holder());
    input_grad->ShareInplaceVersionCounterWith(out_grad);
  } else {
    DenseTensor contiguous_tmp;
    DenseTensor tmp_out_grad = out_grad;

    contiguous_tmp.set_meta(tmp_out_grad.meta());

    PD_VISIT_ALL_TYPES(out_grad.dtype(), "ViewShapeGradKernel", ([&] {
                         phi::StridedTensorContiguous<data_t>(tmp_out_grad,
                                                              &contiguous_tmp);
                       }));

    input_grad->set_meta(contiguous_tmp.meta());
    input_grad->Resize(target_dims);
    input_grad->set_strides(DenseTensorMeta::calc_strides(target_dims));
    input_grad->set_offset(0);
    input_grad->ResetHolder(contiguous_tmp.Holder());
  }
}

template <typename Context>
void ViewDtypeGradKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& out_grad,
                         DataType dtype,
                         DenseTensor* input_grad) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  ViewDtypeKernel<Context>(dev_ctx, out_grad, input.dtype(), input_grad);
}
}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(view_shape_grad,
                                         STRIDED,
                                         phi::ViewShapeGradKernel) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(view_dtype_grad,
                                         STRIDED,
                                         phi::ViewDtypeGradKernel) {}
