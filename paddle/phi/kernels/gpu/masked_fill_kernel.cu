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

#include "paddle/phi/kernels/masked_fill_kernel.h"
#include "paddle/phi/kernels/funcs/masked_fill_utils.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {

template <typename T, int VecSize>
__global__ void GPUMaskedFillOneValueKernel(const T* input,
                                            const bool* mask,
                                            const T* value,
                                            const int64_t input_len,
                                            const int64_t batch_size,
                                            T* output) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= (input_len / VecSize)) {
    return;
  }

  int64_t vec_idx = idx * VecSize;
  int64_t mask_idx = vec_idx / batch_size;
  using VecType = kps::details::VectorType<T, VecSize>;
  const VecType* src = reinterpret_cast<const VecType*>(&input[vec_idx]);
  VecType* dst = reinterpret_cast<VecType*>(&output[vec_idx]);

  T set_value[VecSize];
#pragma unroll
  for (int i = 0; i < VecSize; i++) {
    set_value[i] = value[0];
  }
  const VecType* vec_value = reinterpret_cast<const VecType*>(&set_value[0]);

  if (mask[mask_idx]) {
    *dst = *vec_value;
  } else {
    *dst = *src;
  }
}

template <typename T, int VecSize>
__global__ void GPUMaskedFillKernel(const T* input,
                                    const bool* mask,
                                    const T* value,
                                    const int64_t input_len,
                                    const int64_t batch_size,
                                    T* output) {
  int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);

  if (idx >= (input_len / VecSize)) {
    return;
  }

  int64_t vec_idx = idx * VecSize;
  int64_t mask_idx = vec_idx / batch_size;
  using VecType = kps::details::VectorType<T, VecSize>;
  const VecType* src = reinterpret_cast<const VecType*>(&input[vec_idx]);
  const VecType* value_src = reinterpret_cast<const VecType*>(&value[vec_idx]);
  VecType* dst = reinterpret_cast<VecType*>(&output[vec_idx]);

  if (mask[mask_idx]) {
    *dst = *value_src;
  } else {
    *dst = *src;
  }
}

template <typename T>
void DispatchMaskFillKernel(const phi::GPUContext& dev_ctx,
                            const T* input,
                            const bool* mask,
                            const T* value,
                            const int64_t input_len,
                            const int64_t batch_size,
                            T* output,
                            int vec_size,
                            const phi::backends::gpu::GpuLaunchConfig& config) {
  auto stream = dev_ctx.stream();
  switch (vec_size) {
#define CASE_VECSIZE(__Vs)                                               \
  case __Vs:                                                             \
    GPUMaskedFillKernel<T, __Vs>                                         \
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>( \
            input, mask, value, input_len, batch_size, output);          \
    break;
    CASE_VECSIZE(1)
    CASE_VECSIZE(2)
    CASE_VECSIZE(4)
#undef CASE_VECSIZE
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported vectorized size: %d", vec_size));
  }
}

template <typename T>
void DispatchMaskFillOneValueKernel(
    const phi::GPUContext& dev_ctx,
    const T* input,
    const bool* mask,
    const T* value,
    const int64_t input_len,
    const int64_t batch_size,
    T* output,
    int vec_size,
    const phi::backends::gpu::GpuLaunchConfig& config) {
  auto stream = dev_ctx.stream();
  switch (vec_size) {
#define CASE_VECSIZE(__Vs)                                               \
  case __Vs:                                                             \
    GPUMaskedFillOneValueKernel<T, __Vs>                                 \
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>( \
            input, mask, value, input_len, batch_size, output);          \
    break;
    CASE_VECSIZE(1)
    CASE_VECSIZE(2)
    CASE_VECSIZE(4)
#undef CASE_VECSIZE
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported vectorized size: %d", vec_size));
  }
}

template <typename T>
void GPUMaskedFill(const phi::GPUContext& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& mask,
                   const DenseTensor& value,
                   DenseTensor* output) {
  const T* input_data = input.data<T>();
  const bool* mask_data = mask.data<bool>();
  dev_ctx.template Alloc<T>(output);
  T* output_data = output->data<T>();
  const T* value_data = value.data<T>();
  int64_t input_len = input.numel();
  int64_t mask_len = mask.numel();
  int batch_size = input_len / mask_len;

  int vec_size = 4;
  vec_size = std::min(phi::GetVectorizedSize(input_data), vec_size);
  vec_size = std::min(phi::GetVectorizedSize(output_data), vec_size);
  while (vec_size > 1 && batch_size % vec_size != 0) {
    vec_size /= 2;
  }

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, input_len, vec_size);

  if (value.numel() == 1) {
    DispatchMaskFillOneValueKernel<T>(dev_ctx,
                                      input_data,
                                      mask_data,
                                      value_data,
                                      input_len,
                                      batch_size,
                                      output_data,
                                      vec_size,
                                      config);
  } else {
    PADDLE_ENFORCE_EQ(value.numel(),
                      input_len,
                      common::errors::InvalidArgument(
                          "value.numel() should equal to input.numel()"
                          "but got value.numel() = %d, input.numel() = %d",
                          value.numel(),
                          input_len));

    DispatchMaskFillKernel<T>(dev_ctx,
                              input_data,
                              mask_data,
                              value_data,
                              input_len,
                              batch_size,
                              output_data,
                              vec_size,
                              config);
  }
}

template <typename T, typename Context>
void MaskedFillKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& mask,
                      const DenseTensor& value,
                      DenseTensor* out) {
  if (x.numel() == 0 || mask.numel() == 0) {
    out->Resize({0});
    dev_ctx.template Alloc<T>(out);
    return;
  }

  const auto& x_dims = x.dims();
  const auto& mask_dims = mask.dims();

  auto expanded_size =
      common::vectorize(phi::funcs::BroadcastTwoDims(x_dims, mask_dims, -1));
  DDim expanded_dims = common::make_ddim(expanded_size);

  bool flag = funcs::CanDispatchMaskFillShortcut(x.dims(), mask.dims());
  if (expanded_dims != x_dims) flag = false;

  DenseTensor value_expand = value;
  if (value.numel() != 1 && value.dims() != expanded_dims) {
    phi::ExpandKernel<T, Context>(
        dev_ctx, value, IntArray(expanded_size), &value_expand);
  }

  if (flag) {
    GPUMaskedFill<T>(dev_ctx, x, mask, value_expand, out);
    return;
  }

  DenseTensor mask_expand;
  DenseTensor x_expand;

  if (mask.dims() != expanded_dims) {
    phi::ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  if (x.dims() != expanded_dims) {
    phi::ExpandKernel<T, Context>(
        dev_ctx, x, IntArray(expanded_size), &x_expand);
  } else {
    x_expand = x;
  }

  out->Resize(expanded_dims);

  GPUMaskedFill<T>(dev_ctx, x_expand, mask_expand, value_expand, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_fill,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaskedFillKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
