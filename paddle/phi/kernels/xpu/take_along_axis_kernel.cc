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

#include "glog/logging.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void TakeAlongAxisKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& index,
                         int axis,
                         DenseTensor* out) {
  out->Resize(index.dims());
  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 0 || index.numel() == 0) return;

  const auto& index_dtype = index.dtype();
  bool index_dtype_match =
      index_dtype == DataType::INT32 || index_dtype == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_dtype_match,
                    true,
                    errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        DataTypeToString(index_dtype),
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64)));

  std::vector<int64_t> x_shape(x.dims().size());
  for (int i = 0; i < x.dims().size(); ++i) {
    x_shape[i] = x.dims()[i];
  }
  std::vector<int64_t> index_shape(index.dims().size());
  for (int i = 0; i < index.dims().size(); ++i) {
    index_shape[i] = index.dims()[i];
  }

  if (x_shape.size() <= 1 && index_shape.size() <= 1) {
    for (int i = x_shape.size(); i < 2; ++i) {
      x_shape.push_back(1);
      index_shape.push_back(1);
    }
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  int r = XPU_SUCCESS;
#ifndef PADDLE_WITH_XPU_PLUGIN
  if (index_dtype == DataType::INT32) {
    r = xpu::gather<XPUType, int>(dev_ctx.x_context(),
                                  reinterpret_cast<const XPUType*>(x.data<T>()),
                                  index.data<int>(),
                                  reinterpret_cast<XPUType*>(out->data<T>()),
                                  x_shape,
                                  index_shape,
                                  axis);
  } else {
    r = xpu::gather<XPUType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        index.data<int64_t>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        x_shape,
        index_shape,
        axis);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");
#else
  if (index_dtype == DataType::INT32) {
    r = xpu::plugin::take_along_axis<XPUType, int>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        index.data<int>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        x_shape,
        index_shape,
        axis);
  } else {
    r = xpu::plugin::take_along_axis<XPUType, int64_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        index.data<int64_t>(),
        reinterpret_cast<XPUType*>(out->data<T>()),
        x_shape,
        index_shape,
        axis);
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "take_along_axis");
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(take_along_axis,
                   XPU,
                   ALL_LAYOUT,
                   phi::TakeAlongAxisKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float) {}
