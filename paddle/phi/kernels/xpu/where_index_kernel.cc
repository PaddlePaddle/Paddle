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

#include "paddle/phi/kernels/where_index_kernel.h"

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void WhereIndexKernel(const Context& dev_ctx,
                      const DenseTensor& condition,
                      DenseTensor* out) {
  const T* cond_data = condition.data<T>();
  auto numel = condition.numel();
  auto dims = condition.dims();
  const int rank = dims.size();

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int* true_num = RAII_GUARD.alloc_l3_or_gm<int32_t>(1);
  int true_num_cpu;
  int ret = xpu::nonzero_count(dev_ctx.x_context(), cond_data, true_num, numel);
  PADDLE_ENFORCE_EQ(
      ret,
      XPU_SUCCESS,
      phi::errors::External(
          "XPU nonzero_count kernel return wrong value[%d %s] in WhereIndex",
          ret,
          XPUAPIErrorMsg[ret]));

  paddle::memory::Copy(phi::CPUPlace(),
                       static_cast<void*>(&true_num_cpu),
                       dev_ctx.GetPlace(),
                       static_cast<void*>(true_num),
                       sizeof(int32_t));

  out->Resize(phi::make_ddim({static_cast<int64_t>(true_num_cpu), rank}));
  auto* out_data = dev_ctx.template Alloc<int64_t>(out);

  if (true_num_cpu == 0) {
    return;
  }

  auto condition_shape = phi::vectorize<int>(dims);
  ret = xpu::where(
      dev_ctx.x_context(), cond_data, out_data, condition_shape, true_num_cpu);
  PADDLE_ENFORCE_EQ(ret,
                    XPU_SUCCESS,
                    phi::errors::External(
                        "XPU masked_select kernel return wrong value[%d %s]",
                        ret,
                        XPUAPIErrorMsg[ret]));
}

}  // namespace phi

PD_REGISTER_KERNEL(
    where_index, XPU, ALL_LAYOUT, phi::WhereIndexKernel, int, bool, float) {}
