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

#include "paddle/phi/kernels/take_along_axis_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void TakeAlongAxisKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& index,
                         int axis,
                         DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (x.numel() == 0 || index.numel() == 0) return;

  const auto& index_type = index.dtype();
  bool index_type_match =
      index_type == DataType::INT32 || index_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        DataTypeToString(index_type),
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64)));

  std::vector<int> xshape(x.dims().size());
  for (int i = 0; i < x.dims().size(); ++i) {
    xshape[i] = x.dims()[i];
  }
  std::vector<int> idxshape(index.dims().size());
  for (int i = 0; i < index.dims().size(); ++i) {
    idxshape[i] = index.dims()[i];
  }

  using XPUType = typename XPUTypeTrait<T>::Type;

  int r = XPU_SUCCESS;
  if (index_type == DataType::INT32) {
    r = xpu::gather_element<XPUType, int>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        index.data<int>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        xshape,
        idxshape,
        axis);
  } else {
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int32_t* index_int_data = RAII_GUARD.alloc_l3_or_gm<int32_t>(index.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(index_int_data);
    const int64_t* index_data = index.data<int64_t>();
    r = xpu::cast<int64_t, int32_t>(
        dev_ctx.x_context(), index_data, index_int_data, index.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");

    r = xpu::gather_element<XPUType, int>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        index_int_data,
        reinterpret_cast<XPUType*>(out->data<T>()),
        xshape,
        idxshape,
        axis);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "take_along_axis");
}

}  // namespace phi

PD_REGISTER_KERNEL(take_along_axis,
                   XPU,
                   ALL_LAYOUT,
                   phi::TakeAlongAxisKernel,
                   phi::dtype::float16,
                   float) {}
