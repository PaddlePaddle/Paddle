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

#include "paddle/phi/kernels/index_sample_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {

template <typename T, typename Context>
void IndexSampleGradKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& index,
                           const DenseTensor& out_grad,
                           DenseTensor* in_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto& index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  XPUType* in_grad_data = ctx.template Alloc<XPUType>(in_grad);
  const XPUType* out_grad_data = out_grad.data<XPUType>();
  auto in_grad_shape = common::vectorize<int64_t>(in_grad->dims());
  auto out_grad_shape = common::vectorize<int64_t>(out_grad.dims());
  auto index_shape = common::vectorize<int64_t>(index.dims());

  int r = xpu::constant(
      ctx.x_context(), in_grad_data, in_grad->numel(), static_cast<XPUType>(0));

  if (index_type == phi::DataType::INT32) {
    const int* index_data = index.data<int>();
    r = xpu::scatter_element(ctx.x_context(),
                             in_grad_data,
                             out_grad_data,
                             index_data,
                             in_grad_data,
                             in_grad_shape,
                             out_grad_shape,
                             index_shape,
                             1,
                             1);
  } else if (index_type == phi::DataType::INT64) {
    const int64_t* index_data = index.data<int64_t>();
    r = xpu::scatter_element(ctx.x_context(),
                             in_grad_data,
                             out_grad_data,
                             index_data,
                             in_grad_data,
                             in_grad_shape,
                             out_grad_shape,
                             index_shape,
                             1,
                             1);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter_element");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    index_sample_grad, XPU, ALL_LAYOUT, phi::IndexSampleGradKernel, float) {}
