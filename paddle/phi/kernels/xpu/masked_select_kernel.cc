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

#include "paddle/phi/kernels/masked_select_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/memory/memcpy.h"

namespace phi {

template <typename T, typename Context>
void MaskedSelectKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& mask,
                        DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto input = &x;
  auto* mask_data = mask.data<bool>();
  auto* input_data = reinterpret_cast<const XPUType*>(input->data<T>());
  auto input_dim = input->dims();
  auto mask_dim = mask.dims();
  PADDLE_ENFORCE_EQ(input_dim,
                    mask_dim,
                    phi::errors::InvalidArgument(
                        "The dim size of input and mask in OP(masked_selected) "
                        "must be equal, but got input dim:(%ld), mask dim: "
                        "(%ld). Please check input "
                        "value.",
                        input_dim,
                        mask_dim));
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int* out_size = RAII_GUARD.alloc_l3_or_gm<int32_t>(1);
  int out_size_cpu;

  PADDLE_ENFORCE_XDNN_SUCCESS(
      xpu::nonzero_count(
          dev_ctx.x_context(), mask_data, out_size, mask.numel()),
      "nonzero_count ");
  paddle::memory::Copy(phi::CPUPlace(),
                       static_cast<void*>(&out_size_cpu),
                       mask.place(),
                       static_cast<void*>(out_size),
                       sizeof(int32_t));

  DDim out_dim{out_size_cpu};
  out->Resize(out_dim);
  auto out_data = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out));

  auto input_shape = vectorize<int>(input_dim);
  auto mask_shape = vectorize<int>(mask_dim);

  PADDLE_ENFORCE_XDNN_SUCCESS(xpu::masked_select(dev_ctx.x_context(),
                                                 input_data,
                                                 mask_data,
                                                 out_data,
                                                 input_shape,
                                                 mask_shape,
                                                 out_size_cpu),
                              "masked_select");
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_select,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaskedSelectKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
