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

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

// Simple CPU-based index_select_grad for double that avoids BLAS dependencies
// that cause compilation issues in XPU kernel files.
template <typename T, typename IndexT>
void IndexSelectGradCPUFallback(const DenseTensor& out_grad,
                                const DenseTensor& index,
                                DenseTensor* x_grad,
                                int dim) {
  const T* input_data = out_grad.data<T>();
  const IndexT* index_data = index.data<IndexT>();
  T* out_data = x_grad->data<T>();

  auto input_dim = out_grad.dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = x_grad->dims();

  // Zero out x_grad
  memset(out_data, 0, sizeof(T) * x_grad->numel());

  int64_t slice_size = 1;
  for (int64_t i = dim + 1; i < input_dim_size; i++) {
    slice_size *= input_dim[i];
  }

  int64_t input_width = slice_size * input_dim[dim];
  int64_t output_width = slice_size * output_dim[dim];

  int64_t outer_nums = 1;
  for (int64_t i = 0; i < dim; i++) {
    outer_nums *= input_dim[i];
  }

  int64_t index_size = index.dims()[0];

  for (int64_t i = 0; i < outer_nums; i++) {
    int64_t input_start_offset = i * input_width;
    int64_t output_start_offset = i * output_width;

    for (int64_t j = 0; j < index_size; j++) {
      IndexT index_value = index_data[j];
      if (index_value < 0) {
        index_value += output_dim[dim];
      }
      auto src = input_data + input_start_offset + j * slice_size;
      auto dst = out_data + output_start_offset + index_value * slice_size;
      for (int64_t k = 0; k < slice_size; k++) {
        dst[k] += src[k];
      }
    }
  }
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

  // Integer data types have zero gradients — fill and return early to avoid
  // xdnn calls that don't support integer accumulation.
  if (std::is_integral<T>::value) {
    Full<T, Context>(dev_ctx, x.dims(), 0, x_grad);
    return;
  }

  // xdnn does not provide index_select_grad for double; fall back to CPU
  // computation with proper cross-device data copies.
  if (std::is_same<T, double>::value) {
    DenseTensor out_grad_cpu;
    Copy(dev_ctx, out_grad, CPUPlace(), true, &out_grad_cpu);

    DenseTensor x_grad_cpu;
    x_grad_cpu.Resize(x_grad->dims());

    auto cpu_place = CPUPlace();
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto* cpu_ctx = static_cast<CPUContext*>(pool.Get(cpu_place));
    cpu_ctx->template Alloc<T>(&x_grad_cpu);

    if (index_type == DataType::INT32) {
      IndexSelectGradCPUFallback<T, int>(out_grad_cpu, index, &x_grad_cpu, dim);
    } else {
      IndexSelectGradCPUFallback<T, int64_t>(
          out_grad_cpu, index, &x_grad_cpu, dim);
    }
    Copy(dev_ctx, x_grad_cpu, dev_ctx.GetPlace(), true, x_grad);
    return;
  }
  using XPUType = typename XPUTypeTrait<T>::Type;
  XPUType* x_grad_data =
      reinterpret_cast<XPUType*>((dev_ctx.template Alloc<T>(x_grad)));
  const XPUType* out_grad_data =
      reinterpret_cast<const XPUType*>(out_grad.data<T>());

  auto out_grad_shape = vectorize<int64_t>(out_grad.dims());
  auto x_grad_shape = vectorize<int64_t>(x_grad->dims());

  int r = 0;
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int byte_times = SizeOf(index_type);

  if (index_type == DataType::INT32) {
    const int* index_data = nullptr;
    int8_t* index_ptr = nullptr;
    if (index.place() == CPUPlace()) {
      index_ptr = RAII_GUARD.alloc_l3_or_gm<int8_t>(byte_times * index.numel());
      PADDLE_ENFORCE_XDNN_NOT_NULL(index_ptr);
      memory_utils::Copy(dev_ctx.GetPlace(),
                         reinterpret_cast<void*>(index_ptr),
                         CPUPlace(),
                         reinterpret_cast<const void*>(index.data<int>()),
                         byte_times * index.numel());
    }
    index_data =
        index_ptr ? reinterpret_cast<const int*>(index_ptr) : index.data<int>();
    r = xpu::index_select_grad<XPUType, int>(dev_ctx.x_context(),
                                             nullptr,
                                             index_data,
                                             out_grad_data,
                                             dim,
                                             x_grad_data,
                                             out_grad_shape,
                                             x_grad_shape);
  } else if (index_type == DataType::INT64) {
    const int64_t* index_data = nullptr;
    int8_t* index_ptr = nullptr;
    if (index.place() == CPUPlace()) {
      index_ptr = RAII_GUARD.alloc_l3_or_gm<int8_t>(byte_times * index.numel());
      PADDLE_ENFORCE_XDNN_NOT_NULL(index_ptr);
      memory_utils::Copy(dev_ctx.GetPlace(),
                         reinterpret_cast<void*>(index_ptr),
                         CPUPlace(),
                         reinterpret_cast<const void*>(index.data<int64_t>()),
                         byte_times * index.numel());
    }
    index_data = index_ptr ? reinterpret_cast<const int64_t*>(index_ptr)
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
