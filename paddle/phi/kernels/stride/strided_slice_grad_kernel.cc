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

#include "paddle/phi/kernels/strided_slice_grad_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/strided_utils.h"
#include "paddle/phi/kernels/strided_slice_kernel.h"
namespace phi {

template <typename Context>
void StridedSliceRawGradStridedKernel(const Context& dev_ctx,
                                      const DenseTensor& x,
                                      const DenseTensor& out_grad,
                                      const std::vector<int>& axes,
                                      const IntArray& starts,
                                      const IntArray& ends,
                                      const IntArray& strides,
                                      const std::vector<int>& infer_flags,
                                      const std::vector<int>& decrease_axis,
                                      DenseTensor* x_grad) {
  dev_ctx.Alloc(x_grad, x_grad->dtype());
  x_grad->set_strides(DenseTensorMeta::calc_strides(x_grad->dims()));
  phi::StridedTensorFill<Context>(x_grad->dtype(),
                                  "StridedSliceRawGradStridedKernel",
                                  dev_ctx,
                                  *x_grad,
                                  0,
                                  x_grad);
  DenseTensor tmp;
  tmp.set_layout(out_grad.layout());
  tmp.set_lod(out_grad.lod());
  tmp.set_type(out_grad.dtype());
  tmp.Resize(out_grad.dims());
  StridedSliceRawStridedKernel<Context>(dev_ctx,
                                        *x_grad,
                                        axes,
                                        starts,
                                        ends,
                                        strides,
                                        infer_flags,
                                        decrease_axis,
                                        &tmp);
  phi::StridedTensorCopy<Context>(out_grad.dtype(),
                                  "StridedSliceRawGradStridedKernel",
                                  dev_ctx,
                                  out_grad,
                                  common::vectorize<int64_t>(tmp.dims()),
                                  common::vectorize<int64_t>(tmp.strides()),
                                  tmp.offset(),
                                  &tmp);
}

template <typename Context>
void StridedSliceGradStridedKernel(const Context& dev_ctx,
                                   const DenseTensor& x,
                                   const DenseTensor& out_grad,
                                   const std::vector<int>& axes,
                                   const IntArray& starts,
                                   const IntArray& ends,
                                   const IntArray& strides,
                                   DenseTensor* x_grad) {
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int> decrease_axis;
  StridedSliceRawGradStridedKernel<Context>(dev_ctx,
                                            x,
                                            out_grad,
                                            axes,
                                            starts,
                                            ends,
                                            strides,
                                            infer_flags,
                                            decrease_axis,
                                            x_grad);
}

}  // namespace phi

#ifndef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    strided_slice_raw_grad, STRIDED, phi::StridedSliceRawGradStridedKernel) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE_EXCEPT_CUSTOM(
    strided_slice_grad, STRIDED, phi::StridedSliceGradStridedKernel) {}
#else
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(
    strided_slice_raw_grad, STRIDED, phi::StridedSliceRawGradStridedKernel) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(strided_slice_grad,
                                         STRIDED,
                                         phi::StridedSliceGradStridedKernel) {}
#endif
