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

#include "paddle/phi/kernels/nonzero_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void NonZeroKernel(const Context& dev_ctx,
                   const DenseTensor& condition,
                   DenseTensor* out) {
  auto numel = condition.numel();
  auto dims = condition.dims();
  const int64_t rank = dims.size();

  using XPUType = typename XPUTypeTrait<T>::Type;

  if (numel == 0) {
    dev_ctx.template Alloc<int64_t>(out);
    return;
  }

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int64_t* true_num = RAII_GUARD.alloc_l3_or_gm<int64_t>(1);
  int64_t* workspace =
      RAII_GUARD.alloc_l3_or_gm<int64_t>(dev_ctx.x_context()->ncluster() * 64);
  auto cond_data = reinterpret_cast<const XPUType*>(condition.data<T>());
  int ret = xpu::nonzero_count<XPUType, int64_t>(
      dev_ctx.x_context(), cond_data, true_num, numel, workspace);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "nonzero_count");

  int64_t true_num_cpu;
  memory_utils::Copy(phi::CPUPlace(),
                     static_cast<void*>(&true_num_cpu),
                     dev_ctx.GetPlace(),
                     static_cast<void*>(true_num),
                     sizeof(int64_t));
  if (std::getenv("XPUSIM_SKIP_RUN") &&
      std::strcmp(std::getenv("XPUSIM_SKIP_RUN"), "1") == 0) {
    VLOG(3) << "WARNING: In the simulator mode, the variable true_num_cpu "
               "stores an uninitialized value. To avoid allocating a memory of "
               "random size, we assign numel to true_num_cpu";
    true_num_cpu = numel;
  }

  out->Resize(common::make_ddim({true_num_cpu, rank}));
  auto* out_data = dev_ctx.template Alloc<int64_t>(out);

  if (true_num_cpu == 0) {
    return;
  }

  auto condition_shape = common::vectorize<int64_t>(dims);
  ret = xpu::nonzero_compute<XPUType, int64_t, int64_t>(dev_ctx.x_context(),
                                                        cond_data,
                                                        out_data,
                                                        condition_shape,
                                                        true_num_cpu,
                                                        workspace);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "nonzero_compute");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    nonzero, XPU, ALL_LAYOUT, phi::NonZeroKernel, int, bool, float, int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
