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

#include "paddle/phi/kernels/masked_fill_grad_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/masked_fill_utils.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_grad_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {
template <typename T, int VecSize>
__global__ void GPUMaskedFillXGradKernel(const T* out_grad,
                                         const bool* mask,
                                         const int64_t input_len,
                                         const int64_t batch_size,
                                         T* x_grad) {
  int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);

  if (idx >= (input_len / VecSize)) {
    return;
  }

  int64_t vec_idx = idx * VecSize;
  int64_t mask_idx = vec_idx / batch_size;
  using VecType = kps::details::VectorType<T, VecSize>;
  const VecType* src = reinterpret_cast<const VecType*>(&out_grad[vec_idx]);
  VecType* x_grad_dst = reinterpret_cast<VecType*>(&x_grad[vec_idx]);
  T set_value[VecSize];
#pragma unroll
  for (int i = 0; i < VecSize; i++) {
    set_value[i] = 0;
  }
  const VecType* vec_value = reinterpret_cast<const VecType*>(&set_value[0]);

  if (mask[mask_idx]) {
    *x_grad_dst = *vec_value;
  } else {
    *x_grad_dst = *src;
  }
}

template <typename T, int VecSize>
__global__ void GPUMaskedFillValueGradKernel(const T* out_grad,
                                             const bool* mask,
                                             const int64_t input_len,
                                             const int64_t batch_size,
                                             T* value_grad) {
  int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x);

  if (idx >= (input_len / VecSize)) {
    return;
  }

  int64_t vec_idx = idx * VecSize;
  int64_t mask_idx = vec_idx / batch_size;
  using VecType = kps::details::VectorType<T, VecSize>;
  const VecType* src = reinterpret_cast<const VecType*>(&out_grad[vec_idx]);
  VecType* value_grad_dst = reinterpret_cast<VecType*>(&value_grad[vec_idx]);
  T set_value[VecSize];
#pragma unroll
  for (int i = 0; i < VecSize; i++) {
    set_value[i] = 0;
  }
  const VecType* vec_value = reinterpret_cast<const VecType*>(&set_value[0]);

  if (mask[mask_idx]) {
    *value_grad_dst = *src;
  } else {
    *value_grad_dst = *vec_value;
  }
}

template <typename T, int VecSize>
__global__ void GPUMaskedFillGradKernel(const T* out_grad,
                                        const bool* mask,
                                        const int64_t input_len,
                                        const int64_t batch_size,
                                        T* x_grad,
                                        T* value_grad) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= (input_len / VecSize)) {
    return;
  }

  int64_t vec_idx = idx * VecSize;
  int64_t mask_idx = vec_idx / batch_size;
  using VecType = kps::details::VectorType<T, VecSize>;
  const VecType* src = reinterpret_cast<const VecType*>(&out_grad[vec_idx]);
  VecType* x_grad_dst = reinterpret_cast<VecType*>(&x_grad[vec_idx]);
  VecType* value_grad_dst = reinterpret_cast<VecType*>(&value_grad[vec_idx]);
  T set_value[VecSize];
#pragma unroll
  for (int i = 0; i < VecSize; i++) {
    set_value[i] = 0;
  }
  const VecType* vec_value = reinterpret_cast<const VecType*>(&set_value[0]);

  if (mask[mask_idx]) {
    *x_grad_dst = *vec_value;
    *value_grad_dst = *src;
  } else {
    *x_grad_dst = *src;
    *value_grad_dst = *vec_value;
  }
}

template <typename T>
void DispatchMaskFillGradKernel(
    const phi::GPUContext& dev_ctx,
    const T* input,
    const bool* mask,
    const int64_t input_len,
    const int64_t batch_size,
    T* x_grad,
    T* value_grad,
    int vec_size,
    const phi::backends::gpu::GpuLaunchConfig& config) {
  auto stream = dev_ctx.stream();
  if (x_grad && value_grad) {
    switch (vec_size) {
#define CASE_VECSIZE(__Vs)                                               \
  case __Vs:                                                             \
    GPUMaskedFillGradKernel<T, __Vs>                                     \
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>( \
            input, mask, input_len, batch_size, x_grad, value_grad);     \
    break;
      CASE_VECSIZE(1)
      CASE_VECSIZE(2)
      CASE_VECSIZE(4)
#undef CASE_VECSIZE
      default:
        PADDLE_THROW(common::errors::Unimplemented(
            "Unsupported vectorized size: %d", vec_size));
    }
  } else if (x_grad) {
    switch (vec_size) {
#define CASE_VECSIZE(__Vs)                                               \
  case __Vs:                                                             \
    GPUMaskedFillXGradKernel<T, __Vs>                                    \
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>( \
            input, mask, input_len, batch_size, x_grad);                 \
    break;
      CASE_VECSIZE(1)
      CASE_VECSIZE(2)
      CASE_VECSIZE(4)
#undef CASE_VECSIZE
      default:
        PADDLE_THROW(common::errors::Unimplemented(
            "Unsupported vectorized size: %d", vec_size));
    }
  } else if (value_grad) {
    switch (vec_size) {
#define CASE_VECSIZE(__Vs)                                               \
  case __Vs:                                                             \
    GPUMaskedFillValueGradKernel<T, __Vs>                                \
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>( \
            input, mask, input_len, batch_size, value_grad);             \
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
}

template <typename T>
void DispatchMaskFillOneValueGradKernel(
    const phi::GPUContext& dev_ctx,
    const T* input,
    const bool* mask,
    const int64_t input_len,
    const int64_t batch_size,
    T* x_grad,
    int vec_size,
    const phi::backends::gpu::GpuLaunchConfig& config) {
  auto stream = dev_ctx.stream();
  if (x_grad) {
    switch (vec_size) {
#define CASE_VECSIZE(__Vs)                                               \
  case __Vs:                                                             \
    GPUMaskedFillXGradKernel<T, __Vs>                                    \
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>( \
            input, mask, input_len, batch_size, x_grad);                 \
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
}

template <typename T>
void GPUMaskedFillGrad(const phi::GPUContext& dev_ctx,
                       const DenseTensor& out_grad,
                       const DenseTensor& mask,
                       DenseTensor* x_grad,
                       DenseTensor* value_grad) {
  const T* out_grad_data = out_grad.data<T>();
  const bool* mask_data = mask.data<bool>();

  T* x_grad_data = nullptr;
  T* value_grad_data = nullptr;

  int64_t input_len = out_grad.numel();
  int64_t mask_len = mask.numel();
  int batch_size = input_len / mask_len;

  int vec_size = 4;
  vec_size = std::min(phi::GetVectorizedSize(out_grad_data), vec_size);
  if (x_grad && x_grad->initialized()) {
    x_grad_data = x_grad->data<T>();
    vec_size = std::min(phi::GetVectorizedSize(x_grad_data), vec_size);
  }

  if (value_grad && value_grad->initialized()) {
    value_grad_data = value_grad->data<T>();
    vec_size = std::min(phi::GetVectorizedSize(value_grad_data), vec_size);
  }

  while (vec_size > 1 && batch_size % vec_size != 0) {
    vec_size /= 2;
  }

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, input_len, vec_size);

  if (value_grad && value_grad->numel() == 1) {
    DispatchMaskFillOneValueGradKernel<T>(dev_ctx,
                                          out_grad_data,
                                          mask_data,
                                          input_len,
                                          batch_size,
                                          x_grad_data,
                                          vec_size,
                                          config);
    if (value_grad) {
      DenseTensor zero_tensor;
      FullLikeKernel<T, phi::GPUContext>(
          dev_ctx, out_grad, Scalar(T(0.0)), out_grad.dtype(), &zero_tensor);
      DenseTensor value_grad_tensor;
      value_grad_tensor.set_meta(out_grad.meta());
      WhereKernel<T, phi::GPUContext>(
          dev_ctx, mask, out_grad, zero_tensor, &value_grad_tensor);
      SumKernel<T, phi::GPUContext>(
          dev_ctx, value_grad_tensor, {1}, out_grad.dtype(), false, value_grad);
    }

  } else {
    DispatchMaskFillGradKernel<T>(dev_ctx,
                                  out_grad_data,
                                  mask_data,
                                  input_len,
                                  batch_size,
                                  x_grad_data,
                                  value_grad_data,
                                  vec_size,
                                  config);
  }
}

template <typename T, typename Context>
void MaskedFillGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& mask,
                          const DenseTensor& value UNUSED,
                          const DenseTensor& out_grad,
                          DenseTensor* x_grad,
                          DenseTensor* v_grad) {
  if (out_grad.numel() == 0 || mask.numel() == 0) {
    if (x_grad != nullptr) {
      x_grad->Resize({0});
      dev_ctx.template Alloc<T>(x_grad);
    }
    if (v_grad != nullptr) {
      v_grad->Resize({0});
      dev_ctx.template Alloc<T>(v_grad);
    }
    return;
  }
  auto out_grad_dims = out_grad.dims();
  auto x_grad_dims = x_grad->dims();
  auto mask_dims = mask.dims();
  DenseTensor mask_expand;
  DenseTensor x_grad_expand;
  DenseTensor v_grad_expand;
  bool expand_x = false;
  bool expand_v = false;
  auto expanded_size =
      common::vectorize(funcs::BroadcastTwoDims(x_grad_dims, mask_dims, -1));
  auto expanded_dims = common::make_ddim(expanded_size);
  bool flag = funcs::CanDispatchMaskFillShortcut(out_grad_dims, mask_dims);
  if (expanded_dims != x_grad_dims) flag = false;
  if (v_grad && v_grad->dims() != expanded_dims && v_grad->numel() != 1)
    flag = false;

  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
  }
  if (v_grad) {
    dev_ctx.template Alloc<T>(v_grad);
  }
  if (flag) {
    GPUMaskedFillGrad<T>(dev_ctx, out_grad, mask, x_grad, v_grad);
    return;
  }

  if (mask.dims() != expanded_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  auto mask_size = mask_expand.numel();
  if (mask_size <= 0) return;

  if (x_grad) {
    if (x_grad->dims() != expanded_dims) {
      x_grad_expand = Empty<T, Context>(dev_ctx, IntArray(expanded_size));
      expand_x = true;
    } else {
      x_grad_expand = *x_grad;
    }
  }

  if (v_grad) {
    if (v_grad->dims() != expanded_dims && v_grad->numel() != 1) {
      v_grad_expand = Empty<T, Context>(dev_ctx, IntArray(expanded_size));
      expand_v = true;
    } else {
      v_grad_expand = *v_grad;
    }
  }

  GPUMaskedFillGrad<T>(
      dev_ctx, out_grad, mask_expand, &x_grad_expand, &v_grad_expand);

  if (expand_x) {
    ExpandGradKernel<T, Context>(
        dev_ctx, x, x_grad_expand, IntArray(expanded_size), x_grad);
  }
  if (expand_v) {
    ExpandGradKernel<T, Context>(
        dev_ctx, value, v_grad_expand, IntArray(expanded_size), v_grad);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(masked_fill_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaskedFillGradKernel,
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
