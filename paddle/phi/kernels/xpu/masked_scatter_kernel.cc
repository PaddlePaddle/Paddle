// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/masked_scatter_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/nonzero_kernel.h"
#include "paddle/phi/kernels/stack_kernel.h"

namespace phi {

template <typename T, typename Context>
void MaskedScatterKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& mask,
                         const DenseTensor& value,
                         DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  if (x.numel() == 0 || mask.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  auto x_dims = x.dims();
  auto mask_dims = mask.dims();
  auto expanded_size =
      vectorize(funcs::BroadcastTwoDims(x_dims, mask_dims, -1));
  DDim expanded_dims = make_ddim(expanded_size);

  DenseTensor mask_expand;
  DenseTensor x_expand;

  if (mask_dims != expanded_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  if (x_dims != expanded_dims) {
    ExpandKernel<T, Context>(dev_ctx, x, IntArray(expanded_size), &x_expand);
  } else {
    x_expand = x;
  }

  out->Resize(expanded_dims);
  dev_ctx.template Alloc<T>(out);

  int64_t total = x_expand.numel();
  int64_t value_numel = value.numel();

  // Count number of True elements in mask using nonzero
  DenseTensor indices_int64;
  NonZeroKernel<bool, Context>(dev_ctx, mask_expand, &indices_int64);

  int64_t mask_count = indices_int64.dims()[0];
  int64_t rank = indices_int64.dims().size() > 1 ? indices_int64.dims()[1] : 1;

  PADDLE_ENFORCE_LE(
      mask_count,
      value_numel,
      common::errors::InvalidArgument(
          "Number of True values in mask (%d) exceeds the number of "
          "elements in value (%d).",
          mask_count,
          value_numel));

  // First copy x to out
  auto* x_data = reinterpret_cast<const XPUType*>(x_expand.data<T>());
  auto* out_data = reinterpret_cast<XPUType*>(out->data<T>());
  int r = xpu::copy<XPUType>(dev_ctx.x_context(), x_data, out_data, total);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

  if (mask_count == 0) {
    return;
  }

  // Get indices and convert to 1-D flat indices
  auto* indices_data = indices_int64.data<int64_t>();
  auto* value_data = reinterpret_cast<const XPUType*>(value.data<T>());

  // For 1-D case, indices are already flat
  // For multi-D case, need to compute flat indices from coordinates
  // flat_idx = sum(idx[i] * stride[i]) for i in [0, rank-1]
  auto mask_shape = vectorize<int64_t>(expanded_dims);

  // Compute strides on CPU
  std::vector<int64_t> strides(rank);
  if (rank > 0) {
    strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * mask_shape[i + 1];
    }
  }

  // Flatten indices: for each [i, j] in indices, compute flat_indices[i]
  // For rank=1: flat_indices[i] = indices[i, 0]
  // For rank>1: flat_indices[i] = sum_j(indices[i,j] * strides[j])
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int64_t* flat_indices = RAII_GUARD.alloc_l3_or_gm<int64_t>(mask_count);

  if (rank == 1) {
    // For 1-D, indices are already flat - just copy them
    r = xpu::copy<int64_t>(dev_ctx.x_context(), indices_data, flat_indices, mask_count);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  } else {
    // For multi-D, compute flat indices: flat_idx = sum(idx[j] * stride[j])
    // Copy strides to XPU
    int64_t* strides_xpu = RAII_GUARD.alloc_l3_or_gm<int64_t>(rank);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       static_cast<void*>(strides_xpu),
                       CPUPlace(),
                       static_cast<void*>(strides.data()),
                       rank * sizeof(int64_t));

    // Initialize flat_indices to 0
    r = xpu::constant<int64_t>(dev_ctx.x_context(), flat_indices, mask_count, 0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

    // For each dimension, add idx * stride to flat_indices
    int64_t* col_indices = RAII_GUARD.alloc_l3_or_gm<int64_t>(mask_count);
    for (int j = 0; j < rank; ++j) {
      // Extract column j: col_indices[i] = indices[i * rank + j]
      // Use strided_slice with stride (rank, 1) starting at j
      r = xpu::strided_slice<int64_t>(dev_ctx.x_context(),
                                        indices_data,
                                        col_indices,
                                        {mask_count * rank},
                                        {j},
                                        {mask_count * rank},
                                        {rank});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_slice");

      // Multiply col_indices by stride[j] and add to flat_indices
      // flat_indices[i] += col_indices[i] * strides[j]
      // We can use broadcast_add after scaling col_indices

      // First, scale col_indices: col_indices = col_indices * strides[j]
      // Use elementwise_mul with a constant stride
      DenseTensor col_scaled(DataType::INT64);
      col_scaled.Resize(phi::make_ddim({mask_count}));
      auto* col_scaled_data = dev_ctx.template Alloc<int64_t>(&col_scaled);

      r = xpu::broadcast_mul<int64_t>(dev_ctx.x_context(),
                                       col_indices,
                                       &strides[j],
                                       col_scaled_data,
                                       {mask_count},
                                       {1});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

      // Add to flat_indices
      r = xpu::broadcast_add<int64_t>(dev_ctx.x_context(),
                                       col_scaled_data,
                                       flat_indices,
                                       flat_indices,
                                       {mask_count},
                                       {mask_count});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
    }
  }

  // Use scatter to write values to output at flat_indices positions
  xpu::VectorParam<int64_t> indices_vec{nullptr, mask_count, flat_indices};

  int64_t dim0 = total;
  int64_t dim1 = 1;

  r = xpu::scatter<XPUType>(dev_ctx.x_context(),
                            value_data,
                            out_data,
                            indices_vec,
                            dim0,
                            dim1,
                            true);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter");
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_scatter,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaskedScatterKernel,
                   float,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
