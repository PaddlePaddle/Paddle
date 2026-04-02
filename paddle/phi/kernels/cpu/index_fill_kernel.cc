// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_fill_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

// CPU implementation of the index_fill core loop.
// Uses the same three-segment decomposition as the GPU kernel:
//   offset = outer * (dim_size * inner_size) + idx * inner_size + inner
//
// Loop order: index (outermost, OMP-parallelized) → outer → inner (innermost).
// Putting the index loop outermost ensures each OMP thread works on independent
// slices. Putting inner loop innermost ensures contiguous memory writes,
// which is cache-friendly on CPU.
template <typename T>
void index_fill_kernel(const int64_t* index_data,
                       const int64_t index_size,
                       const T fill_value,
                       const int64_t outer_size,
                       const int64_t dim_size,
                       const int64_t inner_size,
                       T* out) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < index_size; ++i) {
    int64_t idx = index_data[i];
    if (idx < 0) {
      idx += dim_size;
    }

    for (int64_t outer = 0; outer < outer_size; ++outer) {
      int64_t base_offset = outer * dim_size * inner_size + idx * inner_size;

      // This innermost loop writes to contiguous memory (good for cache).
      for (int64_t inner = 0; inner < inner_size; ++inner) {
        out[base_offset + inner] = fill_value;
      }
    }
  }
}

// CPU host-side launch function.
template <typename T, typename Context>
void LaunchIndexFillKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& index,
                           int axis,
                           const T fill_value,
                           DenseTensor* out) {
  const T* x_data = x.data<T>();
  const int64_t numel = x.numel();
  bool is_initialized = out->initialized();

  T* out_data = dev_ctx.template Alloc<T>(out);

  // Copy-then-modify: copy x to out first, skip if already sharing memory
  // (inplace).
  if (!is_initialized || (x.data<T>() != out->data<T>())) {
    std::memcpy(out_data, x_data, numel * sizeof(T));
  }

  if (index.numel() == 0) {
    return;
  }

  // Cast int32 index to int64 on CPU (simple loop, no GPU kernel needed).
  DenseTensor index_int64;
  const DenseTensor* ptr_index = nullptr;

  if (index.dtype() == DataType::INT32) {
    index_int64.Resize(index.dims());
    int64_t* index_int64_data = dev_ctx.template Alloc<int64_t>(&index_int64);
    const int32_t* index_int32_data = index.data<int32_t>();

    int64_t index_numel = index.numel();
    for (int64_t i = 0; i < index_numel; ++i) {
      index_int64_data[i] = static_cast<int64_t>(index_int32_data[i]);
    }

    ptr_index = &index_int64;
  } else {
    ptr_index = &index;
  }

  const int64_t* index_data = ptr_index->data<int64_t>();
  const int64_t index_size = ptr_index->numel();

  // Three-segment decomposition: split dims around the target axis.
  const auto& x_dims = x.dims();
  const int64_t x_dims_size = x_dims.size();

  int64_t outer_size = 1;  // product of dims before axis
  for (int64_t i = 0; i < axis; ++i) {
    outer_size *= x_dims[i];
  }

  int64_t axis_size = x_dims[axis];  // the target dimension size

  int64_t inner_size = 1;  // product of dims after axis
  for (int64_t i = axis + 1; i < x_dims_size; ++i) {
    inner_size *= x_dims[i];
  }

  index_fill_kernel<T>(index_data,
                       index_size,
                       fill_value,
                       outer_size,
                       axis_size,
                       inner_size,
                       out_data);
}

template <typename T, typename Context>
void IndexFillKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& index,
                     int axis,
                     const Scalar& value,
                     DenseTensor* out) {
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const int64_t x_dims_size = x.dims().size();

  if (axis < 0) {
    axis += x_dims_size;
  }

  T fill_value = value.to<T>();

  LaunchIndexFillKernel<T, Context>(dev_ctx, x, index, axis, fill_value, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_fill,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexFillKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
