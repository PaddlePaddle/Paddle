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

#include "paddle/phi/kernels/slice_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {

template <typename T, typename Context>
void ScatterGradRawKernel(const Context &ctx,
                        const DenseTensor &index,
                        const DenseTensor &updates,
                        const DenseTensor &out_grad,
                        bool overwrite,
                        DenseTensor *x_grad,
                        DenseTensor *updates_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  
  auto index_type = index.dtype();
  int64_t index_size = static_cast<int64_t>(index.dims()[0]);
  bool index_type_match =
    index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;

  if (!index_type_match) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "scatter_op Index holds the wrong type, it holds [%s],", index_type));
  }

  const XPUType* dy_ptr = reinterpret_cast<const XPUType*>(out_grad.data<T>());
  XPUType* dx_ptr = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(x_grad));
  XPUType* dpdates_ptr = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(updates_grad));

//   (void)index_ptr;
//   (void)dx_ptr;
//   (void)dpdates_ptr;

  std::vector<int> xshape(out_grad.dims().size());
  for (int i = 0; i < out_grad.dims().size(); ++i) {
    xshape[i] = out_grad.dims()[i];
  }

  int r = 0;
  if (index_type == phi::DataType::INT32) {
    auto index_ptr = const_cast<int *>(index.data<int>());
    xpu::VectorParam<int> indices{nullptr, index_size, index_ptr};
    r = xpu::scatter_grad(ctx.x_context(), 
                            dy_ptr, 
                            indices, 
                            dx_ptr, 
                            dpdates_ptr,
                            xshape, 
                            overwrite);
  } else {
    auto index_ptr = const_cast<int64_t *>(index.data<int64_t>());
    xpu::VectorParam<int64_t> indices{nullptr, index_size, index_ptr};
    r = xpu::scatter_grad(ctx.x_context(), 
                            dy_ptr, 
                            indices, 
                            dx_ptr, 
                            dpdates_ptr,
                            xshape, 
                            overwrite);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(scatter_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ScatterGradRawKernel,
                   float,
                   phi::dtype::float16) {}
