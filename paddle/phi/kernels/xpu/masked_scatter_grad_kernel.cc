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

#include "paddle/phi/kernels/masked_scatter_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/masked_select_kernel.h"
#include "paddle/phi/kernels/nonzero_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void MaskedScatterGradKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& mask,
                             const DenseTensor& value,
                             const DenseTensor& out_grad,
                             DenseTensor* x_grad,
                             DenseTensor* value_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  if (out_grad.numel() == 0 || mask.numel() == 0) {
    if (x_grad) {
      phi::Full<T, Context>(dev_ctx,
                            phi::IntArray(common::vectorize(x_grad->dims())),
                            static_cast<T>(0),
                            x_grad);
    }
    if (value_grad) {
      phi::Full<T, Context>(
          dev_ctx,
          phi::IntArray(common::vectorize(value_grad->dims())),
          static_cast<T>(0),
          value_grad);
    }
    return;
  }

  auto out_grad_dims = out_grad.dims();
  auto mask_dims = mask.dims();
  auto expanded_size =
      vectorize(funcs::BroadcastTwoDims(out_grad_dims, mask_dims, -1));
  DDim expanded_dims = make_ddim(expanded_size);

  DenseTensor mask_expand;
  if (mask_dims != expanded_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  auto* mask_data = mask_expand.data<bool>();
  auto* out_grad_data = reinterpret_cast<const XPUType*>(out_grad.data<T>());
  int64_t total = out_grad.numel();

  if (x_grad) {
    auto x_grad_dims = x_grad->dims();
    if (x_grad_dims == out_grad_dims) {
      // No broadcast happened, compute directly into x_grad.
      dev_ctx.template Alloc<T>(x_grad);
      auto* x_grad_data = reinterpret_cast<XPUType*>(x_grad->data<T>());

      // x_grad = out_grad where mask is False, 0 where mask is True
      // For each element: x_grad[i] = mask[i] ? 0 : out_grad[i]
      // Since we can't use masked_fill directly, we do this manually:
      // First, copy out_grad to x_grad
      int r = xpu::copy<XPUType>(dev_ctx.x_context(), out_grad_data, x_grad_data, total);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

      // Get indices where mask is True and set corresponding positions to 0
      DenseTensor true_indices;
      NonZeroKernel<bool, Context>(dev_ctx, mask_expand, &true_indices);
      int64_t true_count = true_indices.dims()[0];

      if (true_count > 0) {
        // Use index_put or scatter to set positions to 0
        // Simpler: create a tensor of zeros and scatter it
        xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
        XPUType* zeros = RAII_GUARD.alloc_l3_or_gm<XPUType>(true_count);
        r = xpu::constant<XPUType>(dev_ctx.x_context(), zeros, true_count, static_cast<XPUType>(0));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

        // Get flat indices
        int64_t rank = true_indices.dims().size() > 1 ? true_indices.dims()[1] : 1;
        auto* indices_data = true_indices.data<int64_t>();

        if (rank == 1) {
          // For 1-D, use scatter directly
          xpu::VectorParam<int64_t> indices_vec{nullptr, true_count, indices_data};
          r = xpu::scatter<XPUType>(dev_ctx.x_context(),
                                    zeros,
                                    x_grad_data,
                                    indices_vec,
                                    total,
                                    1,
                                    true);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter");
        } else {
          // For multi-D, compute flat indices
          auto mask_shape = vectorize<int64_t>(expanded_dims);
          std::vector<int64_t> strides(rank);
          if (rank > 0) {
            strides[rank - 1] = 1;
            for (int i = rank - 2; i >= 0; --i) {
              strides[i] = strides[i + 1] * mask_shape[i + 1];
            }
          }

          int64_t* flat_indices = RAII_GUARD.alloc_l3_or_gm<int64_t>(true_count);
          int64_t* strides_xpu = RAII_GUARD.alloc_l3_or_gm<int64_t>(rank);
          memory_utils::Copy(dev_ctx.GetPlace(),
                           static_cast<void*>(strides_xpu),
                           CPUPlace(),
                           static_cast<void*>(strides.data()),
                           rank * sizeof(int64_t));

          r = xpu::constant<int64_t>(dev_ctx.x_context(), flat_indices, true_count, 0);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

          int64_t* col_indices = RAII_GUARD.alloc_l3_or_gm<int64_t>(true_count);
          for (int j = 0; j < rank; ++j) {
            r = xpu::strided_slice<int64_t>(dev_ctx.x_context(),
                                              indices_data,
                                              col_indices,
                                              {true_count * rank},
                                              {j},
                                              {true_count * rank},
                                              {rank});
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_slice");

            DenseTensor col_scaled(DataType::INT64);
            col_scaled.Resize(phi::make_ddim({true_count}));
            auto* col_scaled_data = dev_ctx.template Alloc<int64_t>(&col_scaled);

            r = xpu::broadcast_mul<int64_t>(dev_ctx.x_context(),
                                             col_indices,
                                             &strides[j],
                                             col_scaled_data,
                                             {true_count},
                                             {1});
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

            r = xpu::broadcast_add<int64_t>(dev_ctx.x_context(),
                                             col_scaled_data,
                                             flat_indices,
                                             flat_indices,
                                             {true_count},
                                             {true_count});
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
          }

          xpu::VectorParam<int64_t> indices_vec{nullptr, true_count, flat_indices};
          r = xpu::scatter<XPUType>(dev_ctx.x_context(),
                                    zeros,
                                    x_grad_data,
                                    indices_vec,
                                    total,
                                    1,
                                    true);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter");
        }
      }
    } else {
      // Broadcast happened: compute at broadcast shape, then reduce-sum.
      DenseTensor x_grad_broadcast;
      x_grad_broadcast.Resize(expanded_dims);
      dev_ctx.template Alloc<T>(&x_grad_broadcast);
      auto* x_grad_broadcast_data = reinterpret_cast<XPUType*>(x_grad_broadcast.data<T>());

      // Copy out_grad
      int r = xpu::copy<XPUType>(dev_ctx.x_context(), out_grad_data, x_grad_broadcast_data, total);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

      // Zero out where mask is True
      DenseTensor true_indices;
      NonZeroKernel<bool, Context>(dev_ctx, mask_expand, &true_indices);
      int64_t true_count = true_indices.dims()[0];

      if (true_count > 0) {
        xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
        XPUType* zeros = RAII_GUARD.alloc_l3_or_gm<XPUType>(true_count);
        r = xpu::constant<XPUType>(dev_ctx.x_context(), zeros, true_count, static_cast<XPUType>(0));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

        int64_t rank = true_indices.dims().size() > 1 ? true_indices.dims()[1] : 1;
        auto* indices_data = true_indices.data<int64_t>();

        if (rank == 1) {
          xpu::VectorParam<int64_t> indices_vec{nullptr, true_count, indices_data};
          r = xpu::scatter<XPUType>(dev_ctx.x_context(),
                                    zeros,
                                    x_grad_broadcast_data,
                                    indices_vec,
                                    total,
                                    1,
                                    true);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter");
        } else {
          auto mask_shape = vectorize<int64_t>(expanded_dims);
          std::vector<int64_t> strides(rank);
          if (rank > 0) {
            strides[rank - 1] = 1;
            for (int i = rank - 2; i >= 0; --i) {
              strides[i] = strides[i + 1] * mask_shape[i + 1];
            }
          }

          int64_t* flat_indices = RAII_GUARD.alloc_l3_or_gm<int64_t>(true_count);
          int64_t* strides_xpu = RAII_GUARD.alloc_l3_or_gm<int64_t>(rank);
          memory_utils::Copy(dev_ctx.GetPlace(),
                           static_cast<void*>(strides_xpu),
                           CPUPlace(),
                           static_cast<void*>(strides.data()),
                           rank * sizeof(int64_t));

          r = xpu::constant<int64_t>(dev_ctx.x_context(), flat_indices, true_count, 0);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

          int64_t* col_indices = RAII_GUARD.alloc_l3_or_gm<int64_t>(true_count);
          for (int j = 0; j < rank; ++j) {
            r = xpu::strided_slice<int64_t>(dev_ctx.x_context(),
                                              indices_data,
                                              col_indices,
                                              {true_count * rank},
                                              {j},
                                              {true_count * rank},
                                              {rank});
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_slice");

            DenseTensor col_scaled(DataType::INT64);
            col_scaled.Resize(phi::make_ddim({true_count}));
            auto* col_scaled_data = dev_ctx.template Alloc<int64_t>(&col_scaled);

            r = xpu::broadcast_mul<int64_t>(dev_ctx.x_context(),
                                             col_indices,
                                             &strides[j],
                                             col_scaled_data,
                                             {true_count},
                                             {1});
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

            r = xpu::broadcast_add<int64_t>(dev_ctx.x_context(),
                                             col_scaled_data,
                                             flat_indices,
                                             flat_indices,
                                             {true_count},
                                             {true_count});
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
          }

          xpu::VectorParam<int64_t> indices_vec{nullptr, true_count, flat_indices};
          r = xpu::scatter<XPUType>(dev_ctx.x_context(),
                                    zeros,
                                    x_grad_broadcast_data,
                                    indices_vec,
                                    total,
                                    1,
                                    true);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter");
        }
      }

      std::vector<int> reduce_dims =
          funcs::GetReduceDim(x_grad_dims, expanded_dims, -1);
      phi::SumKernel<T, Context>(dev_ctx,
                                 x_grad_broadcast,
                                 reduce_dims,
                                 x_grad_broadcast.dtype(),
                                 false,
                                 x_grad);
    }
  }

  if (value_grad) {
    dev_ctx.template Alloc<T>(value_grad);
    auto* value_grad_data = reinterpret_cast<XPUType*>(value_grad->data<T>());
    int64_t value_numel = value_grad->numel();

    // Initialize value_grad to 0
    int r = xpu::constant<XPUType>(dev_ctx.x_context(), value_grad_data, value_numel, static_cast<XPUType>(0));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

    // Use masked_select to get out_grad elements where mask is True
    DenseTensor value_grad_selected;
    MaskedSelectKernel<T, Context>(dev_ctx, out_grad, mask_expand, &value_grad_selected);

    // value_grad_selected has shape [count], we need to copy it to value_grad
    int64_t selected_count = value_grad_selected.numel();
    auto* selected_data = reinterpret_cast<const XPUType*>(value_grad_selected.data<T>());

    // Copy selected values to value_grad
    r = xpu::copy<XPUType>(dev_ctx.x_context(), selected_data, value_grad_data, selected_count);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_scatter_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaskedScatterGradKernel,
                   float,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
