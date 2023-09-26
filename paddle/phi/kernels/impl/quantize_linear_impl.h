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

#pragma once

#include <string>

#include "paddle/phi/kernels/quantize_linear_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/cast_kernel.h"

namespace phi {

template <typename T>
__global__ void KeDequantize(
    const T* in, const T* scale, T max_range, int64_t num, T* out) {
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int64_t i = idx; i < num; i += blockDim.x * gridDim.x) {
    out[i] = in[i] * scale[0] / max_range;
  }
}

template <typename T>
__global__ void DequantizeOneScaleQuantAxisN(const T* in,
                                             const T* scale,
                                             const T max_range,
                                             const int64_t num,
                                             const int n_scales,
                                             const int quant_stride,
                                             T* out) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int64_t i = idx; i < num; i += blockDim.x * gridDim.x) {
    T s = scale[(i / quant_stride) % n_scales];
    out[i] = in[i] * s / max_range;
  }
}

template <typename DeviceContext, typename T>
struct DequantizeFunctor {
  void operator()(const DeviceContext& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor* scale,
                  T max_range,
                  phi::DenseTensor* out);
};

template <typename T>
struct DequantizeFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor* scale,
                  T max_range,
                  phi::DenseTensor* out) {
    const T* in_data = in->data<T>();
    const T* scale_factor = scale->data<T>();
    T* out_data = dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));

    int64_t num = in->numel();
    int64_t block_size = std::min(
        num, static_cast<int64_t>(dev_ctx.GetMaxThreadsPerBlock() / 4));
    int64_t max_threads =
        dev_ctx.GetMaxPhysicalThreadCount();  // SM * block_per_SM
    const int64_t max_blocks =
        std::max(((max_threads - 1) / block_size + 1), static_cast<int64_t>(1));
    const int64_t grid_size =
        std::min(max_blocks, (num + block_size - 1) / block_size);
    KeDequantize<T><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
        in_data, scale_factor, max_range, num, out_data);
  }
};

template <typename DeviceContext, typename T>
struct ChannelDequantizeFunctorV2 {
  void operator()(const DeviceContext& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor** scales,
                  const int scale_num,
                  T max_range,
                  const int quant_axis,
                  phi::DenseTensor* out);
};

template <typename T>
struct ChannelDequantizeFunctorV2<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor* scale,
                  T max_range,
                  const int quant_axis,
                  phi::DenseTensor* out) {
    auto in_dims = in->dims();
    const T* in_data = in->data<T>();
    T* out_data = dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));
    int64_t num = in->numel();
    const T* scale_factor = scale->data<T>();
    int64_t block_size = std::min(
        num, static_cast<int64_t>(dev_ctx.GetMaxThreadsPerBlock() / 4));
    int64_t max_threads =
        dev_ctx.GetMaxPhysicalThreadCount();  // SM * block_per_SM
    const int64_t max_blocks =
        std::max(((max_threads - 1) / block_size + 1), static_cast<int64_t>(1));
    const int64_t grid_size =
        std::min(max_blocks, (num + block_size - 1) / block_size);

    int quant_stride = 1;
    for (int i = quant_axis + 1; i < in_dims.size(); i++) {
      quant_stride *= in_dims[i];
    }

    DequantizeOneScaleQuantAxisN<T>
        <<<grid_size, block_size, 0, dev_ctx.stream()>>>(in_data,
                                                         scale_factor,
                                                         max_range,
                                                         num,
                                                         in_dims[quant_axis],
                                                         quant_stride,
                                                         out_data);
  }
};

template struct DequantizeFunctor<phi::GPUContext, phi::dtype::float16>;
template struct DequantizeFunctor<phi::GPUContext, float>;
template struct DequantizeFunctor<phi::GPUContext, double>;
template struct ChannelDequantizeFunctorV2<phi::GPUContext, float16>;
template struct ChannelDequantizeFunctorV2<phi::GPUContext, float>;
template struct ChannelDequantizeFunctorV2<phi::GPUContext, double>;

template <typename T, typename Context, typename D>
void DeQuantizeLinearImpl(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& scale,
                          int quant_axis,
                          int bit_length,
                          bool only_observer,
                          DenseTensor* out) {
  auto* in = &x;

  auto in_tmp = phi::Cast<T>(dev_ctx, *in, phi::CppTypeToDataType<D>::Type());

  dev_ctx.template Alloc<D>(out, out->numel() * sizeof(D));

  if (only_observer) {
    phi::Copy(dev_ctx, *in, dev_ctx.GetPlace(), false, out);
    return;
  }

  if (quant_axis < 0) {
    float max_range = (std::pow(2, bit_length - 1) - 1);
    DequantizeFunctor<Context, D>()(
        dev_ctx, &in_tmp, &scale, static_cast<D>(max_range), out);
  } else {
    PADDLE_ENFORCE_EQ(
        scale.numel(),
        in_tmp.dims()[quant_axis],
        phi::errors::PreconditionNotMet(
            "The number of first scale values must be the same with "
            "quant_axis dimension value of Input(X) when the `scale` has "
            "only one element, but %ld != %ld here.",
            scale.numel(),
            in_tmp.dims()[quant_axis]));
    int max_range = (std::pow(2, bit_length - 1) - 1);

    ChannelDequantizeFunctorV2<Context, D>()(
        dev_ctx, &in_tmp, &scale, static_cast<D>(max_range), quant_axis, out);
  }
}

template <typename T, typename Context>
void DeQuantizeLinearKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& scale,
                            const DenseTensor& zero_point,
                            const paddle::optional<DenseTensor>& in_accum,
                            const paddle::optional<DenseTensor>& in_state,
                            int quant_axis,
                            int bit_length,
                            int round_type,
                            bool is_test,
                            bool only_observer,
                            DenseTensor* out,
                            DenseTensor* out_state,
                            DenseTensor* out_accum,
                            DenseTensor* out_scale) {
  switch (scale.dtype()) {
    case phi::DataType::FLOAT64:
      DeQuantizeLinearImpl<T, Context, double>(
          dev_ctx, x, scale, quant_axis, bit_length, only_observer, out);
      break;
    case phi::DataType::FLOAT32:
      DeQuantizeLinearImpl<T, Context, float>(
          dev_ctx, x, scale, quant_axis, bit_length, only_observer, out);
      break;
    case phi::DataType::FLOAT16:
      DeQuantizeLinearImpl<T, Context, float16>(
          dev_ctx, x, scale, quant_axis, bit_length, only_observer, out);
      break;
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "In DeQuantizeLinearKernel, "
          "data type %d for scale/output is not supported ",
          scale.dtype()));
      break;
  }
}

}  // namespace phi
