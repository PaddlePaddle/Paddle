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

#include "paddle/phi/kernels/index_select_grad_kernel.h"

#include <algorithm>
#include <type_traits>
#include <vector>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename Context, typename IndexT>
void ValidateIndexSelectGradIndex(const Context& dev_ctx,
                                  const DenseTensor& index,
                                  int64_t axis_dim) {
  const IndexT* index_data = nullptr;
  std::vector<IndexT> index_cpu_data;
  if (index.place().GetType() == AllocationType::CPU) {
    index_data = index.data<IndexT>();
  } else {
    index_cpu_data.resize(index.numel());
    memory_utils::Copy(CPUPlace(),
                       index_cpu_data.data(),
                       dev_ctx.GetPlace(),
                       index.data<IndexT>(),
                       sizeof(IndexT) * index.numel());
    index_data = index_cpu_data.data();
  }

  for (int64_t i = 0; i < index.numel(); ++i) {
    PADDLE_ENFORCE_GE(
        index_data[i],
        -axis_dim,
        common::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= %ld and < %ld, but got %ld. Please check input "
            "value.",
            -axis_dim,
            axis_dim,
            static_cast<int64_t>(index_data[i])));
    PADDLE_ENFORCE_LT(
        index_data[i],
        axis_dim,
        common::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= %ld and < %ld, but got %ld. Please check input "
            "value.",
            -axis_dim,
            axis_dim,
            static_cast<int64_t>(index_data[i])));
  }
}

template <typename T, typename Context, typename IndexT>
void IndexSelectGradFallback(const Context& dev_ctx,
                             const DenseTensor& index,
                             const DenseTensor& out_grad,
                             int dim,
                             DenseTensor* x_grad) {
  ValidateIndexSelectGradIndex<Context, IndexT>(
      dev_ctx, index, x_grad->dims()[dim]);

  std::vector<T> out_grad_cpu(out_grad.numel());
  memory_utils::Copy(CPUPlace(),
                     out_grad_cpu.data(),
                     out_grad.place(),
                     out_grad.data<T>(),
                     sizeof(T) * out_grad.numel());

  std::vector<IndexT> index_cpu(index.numel());
  if (index.place().GetType() == AllocationType::CPU) {
    std::copy(index.data<IndexT>(),
              index.data<IndexT>() + index.numel(),
              index_cpu.data());
  } else {
    memory_utils::Copy(CPUPlace(),
                       index_cpu.data(),
                       index.place(),
                       index.data<IndexT>(),
                       sizeof(IndexT) * index.numel());
  }

  std::vector<T> x_grad_cpu(x_grad->numel(), static_cast<T>(0));
  const auto input_dim = out_grad.dims();
  const auto output_dim = x_grad->dims();

  int64_t slice_size = 1;
  for (auto i = dim + 1; i < input_dim.size(); ++i) {
    slice_size *= input_dim[i];
  }
  const int64_t input_width = slice_size * input_dim[dim];
  const int64_t output_width = slice_size * output_dim[dim];
  int64_t outer_nums = 1;
  for (auto i = 0; i < dim; ++i) {
    outer_nums *= input_dim[i];
  }

  for (int64_t i = 0; i < outer_nums; ++i) {
    const int64_t input_start_offset = i * input_width;
    const int64_t output_start_offset = i * output_width;
    for (int64_t j = 0; j < index.numel(); ++j) {
      IndexT index_value = index_cpu[j];
      if (index_value < 0) {
        index_value += output_dim[dim];
      }
      const T* src = out_grad_cpu.data() + input_start_offset + j * slice_size;
      T* dst =
          x_grad_cpu.data() + output_start_offset + index_value * slice_size;
      for (int64_t k = 0; k < slice_size; ++k) {
        dst[k] = static_cast<T>(dst[k] + src[k]);
      }
    }
  }

  T* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  memory_utils::Copy(dev_ctx.GetPlace(),
                     x_grad_data,
                     CPUPlace(),
                     x_grad_cpu.data(),
                     sizeof(T) * x_grad->numel());
}

template <typename T, typename Context>
void IndexSelectGradKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& index,
                           const DenseTensor& out_grad,
                           int dim,
                           DenseTensor* x_grad) {
  if (out_grad.numel() == 0) {
    Full<T, Context>(dev_ctx, x.dims(), 0, x_grad);
    return;
  }
  if (dim < 0) {
    dim += out_grad.dims().size();
  }
  const auto& index_type = index.dtype();
  bool index_type_match =
      index_type == DataType::INT32 || index_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        index_type,
                        DataType::INT32,
                        DataType::INT64));

  if constexpr (std::is_same<T, float>::value ||
                std::is_same<T, phi::float16>::value ||
                std::is_same<T, phi::bfloat16>::value) {
    using XPUType = typename XPUTypeTrait<T>::Type;
    XPUType* x_grad_data =
        reinterpret_cast<XPUType*>((dev_ctx.template Alloc<T>(x_grad)));
    const XPUType* out_grad_data =
        reinterpret_cast<const XPUType*>(out_grad.data<T>());

    auto out_grad_shape = vectorize<int64_t>(out_grad.dims());
    auto x_grad_shape = vectorize<int64_t>(x_grad->dims());

    int r = 0;
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int8_t* index_ptr = nullptr;
    int byte_times = SizeOf(index_type);
    if (index.place() == CPUPlace()) {
      index_ptr = RAII_GUARD.alloc_l3_or_gm<int8_t>(byte_times * index.numel());
      PADDLE_ENFORCE_XDNN_NOT_NULL(index_ptr);
      const void* cpu_idx_data = nullptr;
      if (index_type == DataType::INT64) {
        cpu_idx_data = reinterpret_cast<const void*>(index.data<int64_t>());
      } else if (index_type == DataType::INT32) {
        cpu_idx_data = reinterpret_cast<const void*>(index.data<int>());
      }
      memory_utils::Copy(dev_ctx.GetPlace(),
                         reinterpret_cast<void*>(index_ptr),
                         CPUPlace(),
                         cpu_idx_data,
                         byte_times * index.numel());
    }
    if (index_type == DataType::INT32) {
      // XDNN may crash on invalid indices, so match CPU validation first.
      ValidateIndexSelectGradIndex<Context, int>(
          dev_ctx, index, x_grad->dims()[dim]);
      const int* index_data = index_ptr
                                  ? reinterpret_cast<const int*>(index_ptr)
                                  : index.data<int>();
      r = xpu::index_select_grad<XPUType, int>(dev_ctx.x_context(),
                                               nullptr,
                                               index_data,
                                               out_grad_data,
                                               dim,
                                               x_grad_data,
                                               out_grad_shape,
                                               x_grad_shape);
    } else if (index_type == DataType::INT64) {
      // XDNN may crash on invalid indices, so match CPU validation first.
      ValidateIndexSelectGradIndex<Context, int64_t>(
          dev_ctx, index, x_grad->dims()[dim]);
      const int64_t* index_data =
          index_ptr ? reinterpret_cast<const int64_t*>(index_ptr)
                    : index.data<int64_t>();
      r = xpu::index_select_grad<XPUType, int64_t>(dev_ctx.x_context(),
                                                   nullptr,
                                                   index_data,
                                                   out_grad_data,
                                                   dim,
                                                   x_grad_data,
                                                   out_grad_shape,
                                                   x_grad_shape);
    }
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "index_select_grad");
  } else {
    if (index_type == DataType::INT32) {
      IndexSelectGradFallback<T, Context, int>(
          dev_ctx, index, out_grad, dim, x_grad);
    } else if (index_type == DataType::INT64) {
      IndexSelectGradFallback<T, Context, int64_t>(
          dev_ctx, index, out_grad, dim, x_grad);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_select_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexSelectGradKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   double,
                   int,
                   int64_t) {}
