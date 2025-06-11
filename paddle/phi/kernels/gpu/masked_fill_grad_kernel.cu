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
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"

namespace phi {

template <typename T, int VecSize>
__global__ void GPUMaskedFillGradKernel(const T* out_grad,
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
  VecType* dst = reinterpret_cast<VecType*>(&x_grad[vec_idx]);

  T set_value[VecSize];
#pragma unroll
  for (int i = 0; i < VecSize; i++) {
    set_value[i] = 0;
  }
  const VecType* vec_value = reinterpret_cast<const VecType*>(&set_value[0]);

  if (mask[mask_idx]) {
    *dst = *vec_value;
  } else {
    *dst = *src;
  }
}

template <typename T>
void DispatchMaskFillGradKernel(
    const phi::GPUContext& dev_ctx,
    const T* input,
    const bool* mask,
    const int64_t input_len,
    const int64_t batch_size,
    T* output,
    int vec_size,
    const phi::backends::gpu::GpuLaunchConfig& config) {
  auto stream = dev_ctx.stream();
  if (vec_size == 4) {
    GPUMaskedFillGradKernel<T, 4>
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
            input, mask, input_len, batch_size, output);
  } else if (vec_size == 2) {
    GPUMaskedFillGradKernel<T, 2>
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
            input, mask, input_len, batch_size, output);
  } else if (vec_size == 1) {
    GPUMaskedFillGradKernel<T, 1>
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
            input, mask, input_len, batch_size, output);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported vectorized size: %d", vec_size));
  }
}

template <typename T>
void GPUMaskedFillGrad(const phi::GPUContext& dev_ctx,
                       const DenseTensor& out_grad,
                       const DenseTensor& mask,
                       const DenseTensor& value UNUSED,
                       DenseTensor* x_grad) {
  const T* input_data = out_grad.data<T>();
  const bool* mask_data = mask.data<bool>();
  dev_ctx.template Alloc<T>(x_grad);
  T* output_data = x_grad->data<T>();

  int64_t input_len = out_grad.numel();
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

  DispatchMaskFillGradKernel<T>(dev_ctx,
                                input_data,
                                mask_data,
                                input_len,
                                batch_size,
                                output_data,
                                vec_size,
                                config);
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
      dev_ctx.template Alloc<double>(v_grad);
    }
    return;
  }

  auto out_grad_dims = out_grad.dims();
  auto x_grad_dims = x_grad->dims();
  auto mask_dims = mask.dims();

  DenseTensor mask_expand;
  DenseTensor x_grad_expand;

  bool expand_x = false;
  auto expanded_size =
      common::vectorize(funcs::BroadcastTwoDims(x_grad_dims, mask_dims, -1));
  auto expanded_dims = common::make_ddim(expanded_size);

  bool flag = funcs::CanDispatchMaskFillShortcut(out_grad_dims, mask_dims);
  if (expanded_dims != x_grad_dims) flag = false;

  if (flag) {
    if (x_grad != nullptr) {
      GPUMaskedFillGrad<T>(dev_ctx, out_grad, mask, value, x_grad);
    }

    if (v_grad != nullptr) {
      std::vector<int> mask_dims(mask.dims().size());
      std::iota(mask_dims.begin(), mask_dims.end(), 0);
      IntArray mask_axis(mask_dims);
      SumKernel<T>(dev_ctx, mask, mask_axis, v_grad->dtype(), false, v_grad);
      ScaleKernel<T>(dev_ctx,
                     *v_grad,
                     (out_grad.numel() / mask.numel()),
                     0.0f,
                     false,
                     v_grad);
    }
    return;
  }

  if (mask.dims() != expanded_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  if (x_grad->dims() != expanded_dims) {
    x_grad_expand = Empty<T, Context>(dev_ctx, IntArray(expanded_size));
    expand_x = true;
  } else {
    x_grad_expand = *x_grad;
  }

  if (x_grad != nullptr) {
    dev_ctx.template Alloc<T>(x_grad);
    auto mask_size = mask_expand.numel();
    if (mask_size <= 0) return;

    DenseTensor* x_grad_tmp = x_grad;
    if (expand_x) {
      x_grad_tmp = &x_grad_expand;
    }

    GPUMaskedFillGrad<T>(dev_ctx, out_grad, mask_expand, value, x_grad_tmp);

    if (expand_x) {
      ExpandGradKernel<T, Context>(
          dev_ctx, x, x_grad_expand, IntArray(expanded_size), x_grad);
    }
  }

  if (v_grad != nullptr) {
    std::vector<int> v_dims(mask_expand.dims().size());
    std::iota(v_dims.begin(), v_dims.end(), 0);
    IntArray v_axis(v_dims);
    SumKernel<T>(dev_ctx, mask_expand, v_axis, v_grad->dtype(), false, v_grad);
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
