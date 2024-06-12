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

#include "paddle/phi/kernels/top_k_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/xpu/xpu_mem_util.h"
namespace phi {

template <typename T, typename Context>
void TopkKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const Scalar& k_scalar,
                int axis,
                bool largest,
                bool sorted,
                DenseTensor* out,
                DenseTensor* indices) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  const auto& in_dims = x.dims();
  const T* in_data = x.data<T>();
  int64_t* indices_data = dev_ctx.template Alloc<int64_t>(indices);
  T* output_data = dev_ctx.template Alloc<T>(out);

  const auto& out_dims = out->dims();

  PADDLE_ENFORCE_EQ(
      sorted,
      true,
      errors::External(
          "XPU API does not support unsorted topk operation currently."
          " Operator will be supported in future update."));
  if (in_dims.size() == 0) {
    int r = xpu::copy<XPUType>(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(x.data<T>()),
                               reinterpret_cast<XPUType*>(out->data<T>()),
                               x.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

    phi::funcs::set_constant(dev_ctx, indices, static_cast<int64_t>(0));

    return;
  }
  if (axis < 0) axis += in_dims.size();

  size_t k = k_scalar.to<int>();
  if (axis + 1 == in_dims.size()) {
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int32_t* indices_int_data =
        RAII_GUARD.alloc_l3_or_gm<int32_t>(indices->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(indices_int_data);

    const size_t row =
        common::product(common::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const size_t col = in_dims[in_dims.size() - 1];
    int r = xpu::sorted_topk<XPUType>(dev_ctx.x_context(),
                                      reinterpret_cast<const XPUType*>(in_data),
                                      reinterpret_cast<XPUType*>(output_data),
                                      indices_int_data,
                                      row,
                                      col,
                                      k,
                                      largest);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sorted_topk");

    r = xpu::cast<int32_t, int64_t>(dev_ctx.x_context(),
                                    (const int32_t*)indices_int_data,
                                    indices_data,
                                    indices->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  } else {
    // do transpose if axis is not the last dim of input
    std::vector<int> trans_axes;
    for (int i = 0; i < axis; i++) {
      trans_axes.emplace_back(i);
    }
    for (int i = axis + 1; i < in_dims.size(); i++) {
      trans_axes.emplace_back(i);
    }
    trans_axes.emplace_back(axis);
    // Get input and output dims for transpose
    DDim trans_dims(in_dims);
    DDim trans_out_dims(out->dims());
    for (size_t i = 0; i < trans_axes.size(); i++) {
      trans_dims[i] = in_dims[trans_axes[i]];
      trans_out_dims[i] = out_dims[trans_axes[i]];
    }

    std::vector<int> x_shape_host(in_dims.size(), 0);
    for (int i = 0; i < in_dims.size(); ++i) {
      x_shape_host[i] = in_dims[i];
    }

    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    XPUType* trans_in_data = RAII_GUARD.alloc_l3_or_gm<XPUType>(x.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(trans_in_data);

    // Transpose and save interval output to trans_in
    int r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                    reinterpret_cast<const XPUType*>(in_data),
                                    trans_in_data,
                                    x_shape_host,
                                    trans_axes);
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      errors::External("XPU API 1st Transpose kernel"
                                       " returns wrong value[%d %s]!",
                                       r,
                                       XPUAPIErrorMsg[r]));

    XPUType* trans_out_data = RAII_GUARD.alloc_l3_or_gm<XPUType>(out->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(trans_out_data);

    int64_t* trans_idx_data = RAII_GUARD.alloc_l3_or_gm<int64_t>(out->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(trans_idx_data);

    int32_t* trans_idx_int32_data =
        RAII_GUARD.alloc_l3_or_gm<int32_t>(out->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(trans_idx_int32_data);
    const size_t row = common::product(
        common::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const size_t col = trans_dims[trans_dims.size() - 1];

    // Do top k on transposed input
    r = xpu::sorted_topk<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(trans_in_data),
        reinterpret_cast<XPUType*>(trans_out_data),
        trans_idx_int32_data,
        row,
        col,
        k,
        largest);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sorted_topk");

    r = xpu::cast<int32_t, int64_t>(dev_ctx.x_context(),
                                    (const int32_t*)trans_idx_int32_data,
                                    trans_idx_data,
                                    indices->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    // Transpose back to original dims
    std::vector<int> trans_back_axes;
    for (int i = 0; i < axis; i++) {
      trans_back_axes.emplace_back(i);
    }
    trans_back_axes.emplace_back(trans_out_dims.size() - 1);
    for (int i = axis; i < trans_out_dims.size() - 1; i++) {
      trans_back_axes.emplace_back(i);
    }

    std::vector<int> trans_out_shape_host(trans_back_axes.size(), 0);
    for (size_t i = 0; i < trans_back_axes.size(); ++i) {
      trans_out_shape_host[i] = trans_out_dims[i];
    }
    r = xpu::transpose<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(trans_out_data),
        reinterpret_cast<XPUType*>(output_data),
        trans_out_shape_host,
        trans_back_axes);
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      errors::External("XPU API 2nd Transpose kernel"
                                       " returns wrong value[%d %s]",
                                       r,
                                       XPUAPIErrorMsg[r]));
    r = xpu::transpose<int64_t>(dev_ctx.x_context(),
                                trans_idx_data,
                                indices_data,
                                trans_out_shape_host,
                                trans_back_axes);
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      errors::External("XPU API 3rd Transpose kernel"
                                       " returns wrong value[%d %s]",
                                       r,
                                       XPUAPIErrorMsg[r]));
  }
}

template <typename T, typename Context>
void TopkV1Kernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& k_scalar,
                  DenseTensor* out,
                  DenseTensor* indices) {
  TopkKernel<T, Context>(dev_ctx, x, k_scalar, -1, true, true, out, indices);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    topk, XPU, ALL_LAYOUT, phi::TopkKernel, float, phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}

PD_REGISTER_KERNEL(
    topk_v1, XPU, ALL_LAYOUT, phi::TopkV1Kernel, float, phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
