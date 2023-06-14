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

#include <climits>

#include "paddle/phi/kernels/unique_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/core/visit_type.h"

namespace phi {

template <typename Context, typename T, typename IndexT>
void XPUFlattenUniqueKernelImpl(const Context& dev_ctx,
                                const DenseTensor& x,
                                bool return_index,
                                bool return_inverse,
                                bool return_counts,
                                bool is_sorted,
                                DenseTensor* out,
                                DenseTensor* indices,
                                DenseTensor* index,
                                DenseTensor* counts) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto* x_data = x.data<T>();
  int64_t x_len = x.numel();
  int r = XPU_SUCCESS;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int64_t unique_len_cpu = 0;
  int64_t* unique_len_xpu = RAII_GUARD.alloc_l3_or_gm<int64_t>(1);
  if (x_len != 0) {
    r = xpu::unique_count<XPUType, IndexT>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x_data),
        unique_len_xpu,
        x_len,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        false);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "unique_count");
    memory_utils::Copy(phi::CPUPlace(),
                       &unique_len_cpu,
                       dev_ctx.GetPlace(),
                       unique_len_xpu,
                       sizeof(int64_t));
  }
  out->Resize(phi::make_ddim({unique_len_cpu}));
  auto* out_data = dev_ctx.template Alloc<T>(out);
  IndexT* indices_data = nullptr;
  if (return_index) {
    indices->Resize(phi::make_ddim({unique_len_cpu}));
    indices_data = dev_ctx.template Alloc<IndexT>(indices);
  }

  IndexT* inverse_data = nullptr;
  if (return_inverse) {
    index->Resize(phi::make_ddim({x_len}));
    inverse_data = dev_ctx.template Alloc<IndexT>(index);
  }

  IndexT* counts_data = nullptr;
  if (return_counts) {
    counts->Resize(phi::make_ddim({unique_len_cpu}));
    counts_data = dev_ctx.template Alloc<IndexT>(counts);
  }
  if (x_len == 0) {
    return;
  }
  r = xpu::unique_compute<XPUType, IndexT>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(x_data),
      reinterpret_cast<XPUType*>(out_data),
      x_len,
      unique_len_cpu,
      indices_data,
      counts_data,
      inverse_data,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      false);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "unique_compute");
}

template <typename T, typename Context>
void UniqueKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  bool return_index,
                  bool return_inverse,
                  bool return_counts,
                  const std::vector<int>& axis,
                  DataType dtype,
                  DenseTensor* out,
                  DenseTensor* indices,
                  DenseTensor* index,
                  DenseTensor* counts) {
  bool is_sorted = true;
  UniqueRawKernel<T, Context>(dev_ctx,
                              x,
                              return_index,
                              return_inverse,
                              return_counts,
                              axis,
                              dtype,
                              is_sorted,
                              out,
                              indices,
                              index,
                              counts);
}

template <typename T, typename Context>
void UniqueRawKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     bool return_index,
                     bool return_inverse,
                     bool return_counts,
                     const std::vector<int>& axis,
                     DataType dtype,
                     bool is_sorted,
                     DenseTensor* out,
                     DenseTensor* indices,
                     DenseTensor* index,
                     DenseTensor* counts) {
  PADDLE_ENFORCE_EQ(
      axis.empty(), true, "XPU do not support unique with axis now.");
  if (dtype == DataType::INT32) {
    PADDLE_ENFORCE_LE(
        x.numel(),
        INT_MAX,
        phi::errors::InvalidArgument(
            "The number of elements in Input(X) should be less than or "
            "equal to INT_MAX, but received num is %d. Please set `dtype` to "
            "int64.",
            x.numel()));
  }

  PD_VISIT_BASE_INTEGRAL_TYPES(dtype, "XPUFlattenUniqueKernelImpl", [&] {
    XPUFlattenUniqueKernelImpl<Context, T, data_t>(dev_ctx,
                                                   x,
                                                   return_index,
                                                   return_inverse,
                                                   return_counts,
                                                   is_sorted,
                                                   out,
                                                   indices,
                                                   index,
                                                   counts);
  });
}

}  // namespace phi

PD_REGISTER_KERNEL(
    unique, XPU, ALL_LAYOUT, phi::UniqueKernel, float, int, int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(
    unique_raw, XPU, ALL_LAYOUT, phi::UniqueRawKernel, float, int, int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
}
