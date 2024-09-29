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

#include "paddle/phi/kernels/c_embedding_grad_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CEmbeddingGradKernel(const Context& dev_ctx,
                          const DenseTensor& w,
                          const DenseTensor& ids,
                          const DenseTensor& out_grad,
                          int64_t start_index,
                          DenseTensor* w_grad) {
  w_grad->Resize(w.dims());
  dev_ctx.template Alloc(w_grad, w.dtype());
  T* table_grad_data = static_cast<T*>(w_grad->data());
  using XPUType = typename XPUTypeTrait<T>::Type;

  size_t table_t_mem_size = w.numel() * phi::SizeOf(w_grad->dtype());
  size_t table_grad_t_mem_size = w_grad->numel() * phi::SizeOf(w_grad->dtype());

  VLOG(10) << "table_dims:" << w.dims()
           << ", table_t memory_size:" << table_t_mem_size
           << ", table_grad_t memory_size:" << table_grad_t_mem_size
           << ", start_index:" << start_index;

  int r = xpu::constant(dev_ctx.x_context(),
                        reinterpret_cast<XPUType*>(table_grad_data),
                        w_grad->numel(),
                        (XPUType)0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
  const T* d_output_data = out_grad.data<T>();

  const int64_t height = w.dims()[0];
  const int64_t width = w.dims()[1];

  const auto& index_type = ids.dtype();
  if (index_type == phi::DataType::INT32) {
    r = xpu::embedding_grad(dev_ctx.x_context(),
                            reinterpret_cast<const XPUType*>(d_output_data),
                            ids.data<int32_t>(),
                            reinterpret_cast<XPUType*>(table_grad_data),
                            height,
                            width,
                            ids.numel(),
                            -1,
                            static_cast<int32_t>(start_index));
  } else if (index_type == phi::DataType::INT64) {
    r = xpu::embedding_grad(dev_ctx.x_context(),
                            reinterpret_cast<const XPUType*>(d_output_data),
                            ids.data<int64_t>(),
                            reinterpret_cast<XPUType*>(table_grad_data),
                            height,
                            width,
                            ids.numel(),
                            -1,
                            static_cast<int64_t>(start_index));
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "XPU c_embedding ids only support int32 or int64."));
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "embedding_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(c_embedding_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::CEmbeddingGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
