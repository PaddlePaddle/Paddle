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

#include "paddle/phi/kernels/stack_grad_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void StackGradKernel(const Context& dev_ctx,
                     const DenseTensor& out_grad,
                     int axis,
                     std::vector<DenseTensor*> x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto og_dims = out_grad.dims();
  int rank = og_dims.size();
  if (axis < 0) {
    axis += rank;
  }
  int64_t n_slices = og_dims[axis];

  struct ValidSlice {
    DenseTensor* dx;
    DDim final_dims;
  };
  std::vector<ValidSlice> valid_slices;
  valid_slices.reserve(x_grad.size());

  for (size_t i = 0; i < x_grad.size(); ++i) {
    DenseTensor* dx_i = x_grad[i];
    if (dx_i == nullptr || dx_i->numel() == 0) {
      continue;
    }
    ValidSlice vs;
    vs.dx = dx_i;
    vs.final_dims = dx_i->dims();
    valid_slices.push_back(vs);
  }

  if (valid_slices.empty()) {
    return;
  }

  int64_t needed_slices = static_cast<int64_t>(valid_slices.size());

  PADDLE_ENFORCE_LE(
      needed_slices,
      n_slices,
      phi::errors::InvalidArgument(
          "Number of valid slices (%ld) exceeds out_grad's dimension (%ld) "
          "along axis %d in stack_grad kernel. Mismatch between forward and "
          "backward shapes.",
          needed_slices,
          n_slices,
          axis));

  std::vector<int> partial_shape = phi::vectorize<int>(og_dims);
  partial_shape[axis] = static_cast<int>(needed_slices);

  std::vector<XPUType*> dx_ptrs;
  dx_ptrs.reserve(needed_slices);

  std::vector<int> dx_dims_list;
  dx_dims_list.reserve(needed_slices);

  for (auto& vs : valid_slices) {
    dev_ctx.template Alloc<T>(vs.dx);
    dx_ptrs.push_back(reinterpret_cast<XPUType*>(vs.dx->template data<T>()));
    dx_dims_list.push_back(1);
  }

  const XPUType* og_data =
      reinterpret_cast<const XPUType*>(out_grad.template data<T>());

  int r = xpu::split<XPUType>(
      dev_ctx.x_context(), og_data, dx_ptrs, partial_shape, dx_dims_list, axis);

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "split in stack_grad op");

  for (auto& vs : valid_slices) {
    vs.dx->Resize(vs.final_dims);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(stack_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::StackGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int,
                   int16_t,
                   int8_t,
                   uint8_t) {}
