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

#include "paddle/phi/kernels/gather_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GatherKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& index,
                  const Scalar& axis,
                  DenseTensor* out) {
  auto axis_v = axis.to<int64_t>();
  if (axis_v < 0) {
    axis_v += static_cast<int64_t>(x.dims().size());
  }
  const auto& index_type = index.dtype();

  dev_ctx.template Alloc<T>(out);
  if (x.numel() == 0 || index.numel() == 0) return;

  const auto index_dims = index.dims();
  if (index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(
        index_dims[1],
        1,
        common::errors::InvalidArgument(
            "The last dim of index should be 1 when it is 2D, but we get %d",
            index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        index_dims.size() == 1 || index_dims.size() == 0,
        true,
        common::errors::InvalidArgument(
            "The index should be 0D, 1D, when it is not 2D, but we get %d",
            index_dims.size()));
  }
  std::vector<int64_t> xshape(x.dims().size());
  for (int i = 0; i < x.dims().size(); ++i) {
    xshape[i] = x.dims()[i];
  }

  using XPUType = typename XPUTypeTrait<T>::Type;

  int64_t index_len = index.dims().size() == 0 ? 1 : index.dims()[0];
  // XPU SDK paddle_gather may exceed grid limits for large index tensors.
  constexpr int64_t kMaxChunkSize = 32768;

  int r = 0;
  if (index_len <= kMaxChunkSize || axis_v != 0) {
    // Small tensor or non-axis=0: single call is fine
    if (index_type == DataType::INT32) {
      r = xpu::paddle_gather<XPUType, int>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          index.data<int>(),
          reinterpret_cast<XPUType*>(out->data<T>()),
          xshape,
          index_len,
          axis_v);
    } else if (index_type == DataType::INT64) {
      r = xpu::paddle_gather<XPUType, int64_t>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(x.data<T>()),
          index.data<int64_t>(),
          reinterpret_cast<XPUType*>(out->data<T>()),
          xshape,
          index_len,
          axis_v);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unsupported index type, expected int32 or int64, but got type %s",
          DataTypeToString(index_type)));
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");
  } else {
    // Large index with axis=0: chunk to avoid XPU grid limit crash.
    // Output for axis=0 has shape [index_len, D1, D2, ...], so each chunk
    // produces a contiguous slice of the output.
    int64_t inner_size = 1;
    auto out_dims = out->dims();
    for (int64_t i = 1; i < out_dims.size(); ++i) {
      inner_size *= out_dims[i];
    }

    for (int64_t start = 0; start < index_len; start += kMaxChunkSize) {
      int64_t chunk_len = std::min(kMaxChunkSize, index_len - start);
      if (index_type == DataType::INT32) {
        r = xpu::paddle_gather<XPUType, int>(
            dev_ctx.x_context(),
            reinterpret_cast<const XPUType*>(x.data<T>()),
            index.data<int>() + start,
            reinterpret_cast<XPUType*>(out->data<T>()) +
                start * inner_size,
            xshape,
            chunk_len,
            axis_v);
      } else if (index_type == DataType::INT64) {
        r = xpu::paddle_gather<XPUType, int64_t>(
            dev_ctx.x_context(),
            reinterpret_cast<const XPUType*>(x.data<T>()),
            index.data<int64_t>() + start,
            reinterpret_cast<XPUType*>(out->data<T>()) +
                start * inner_size,
            xshape,
            chunk_len,
            axis_v);
      }
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather chunk");
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(gather,
                   XPU,
                   ALL_LAYOUT,
                   phi::GatherKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t) {}
