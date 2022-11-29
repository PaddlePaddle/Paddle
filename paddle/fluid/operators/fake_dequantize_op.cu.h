/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_FLUID_OPERATORS_FAKE_DEQUANTIZE_OP_CU_H_
#define PADDLE_FLUID_OPERATORS_FAKE_DEQUANTIZE_OP_CU_H_
#endif  // PADDLE_FLUID_OPERATORS_FAKE_DEQUANTIZE_OP_CU_H_

#include "paddle/fluid/operators/fake_dequantize_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void KeDequantize(
    const T* in, const T* scale, T max_range, int64_t num, T* out) {
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int64_t i = idx; i < num; i += blockDim.x * gridDim.x) {
    out[i] = in[i] * scale[0] / max_range;
  }
}

template <typename T>
struct DequantizeFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor* scale,
                  T max_range,
                  phi::DenseTensor* out) {
    const T* in_data = in->data<T>();
    const T* scale_factor = scale->data<T>();
    T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());

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

template <typename T>
__global__ void DequantizeOneScaleQuantAxis0(
    const T* in, const T* scale, T max_range, int num, int channel, T* out) {
  int tid = threadIdx.x;
  int channel_size = num / channel;
  const T* in_c = in + blockIdx.x * channel_size;
  T* out_c = out + blockIdx.x * channel_size;
  for (int i = tid; i < channel_size; i += blockDim.x) {
    out_c[i] = in_c[i] * scale[blockIdx.x] / max_range;
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

template <typename T>
__global__ void DequantizeTwoScale(const T* in,
                                   const T* scale_one,
                                   const T* scale_two,
                                   T max_range,
                                   int num,
                                   int n_scales,
                                   int quant_stride,
                                   T* out) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int64_t i = idx; i < num; i += blockDim.x * gridDim.x) {
    int scale_index = (i / quant_stride) % n_scales;
    T s = scale_one[scale_index] * scale_two[0];
    out[i] = in[i] * s / max_range;
  }
}

template <typename T>
struct ChannelDequantizeFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor** scales,
                  const int scale_num,
                  T max_range,
                  const int quant_axis,
                  const int x_num_col_dims,
                  phi::DenseTensor* out) {
    auto in_dims = in->dims();
    const T* in_data = in->data<T>();
    T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());
    if (scale_num == 1) {
      // Dequantize inputs or weights before quantizable operators and after
      // quantization operators. inputs --> quant -- > deqaunt --> conv2d -->
      int64_t num = in->numel();
      const T* scale_factor = scales[0]->data<T>();
      int64_t block_size = std::min(
          num, static_cast<int64_t>(dev_ctx.GetMaxThreadsPerBlock() / 4));
      int64_t max_threads =
          dev_ctx.GetMaxPhysicalThreadCount();  // SM * block_per_SM
      const int64_t max_blocks = std::max(((max_threads - 1) / block_size + 1),
                                          static_cast<int64_t>(1));
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
    } else if (scale_num == 2) {
      // Dequantize activations after quantizable operators.
      // inputs --> quant --> conv2d --> deqaunt -->
      // Note 1:  Not need to consider 'quant_axis'. Because 'quant_aixs' is the
      // axis of weights to be quantized on while dequantization is applied on
      // activations. Note 2: 'x_num_col_dims' is the axis of activations to be
      // quantized on. `x_num_col_dims` is -1 for operator in ['matmul',
      // 'matmul_v2', 'mul'] and is 1 for other operators.
      int64_t num = in->numel();
      int n_scales = in->dims()[x_num_col_dims];
      const T* scale_one = scales[0]->data<T>();
      const T* scale_two = scales[1]->data<T>();

      int64_t block_size = std::min(
          num, static_cast<int64_t>(dev_ctx.GetMaxThreadsPerBlock() / 4));
      int64_t max_threads =
          dev_ctx.GetMaxPhysicalThreadCount();  // SM * block_per_SM
      const int64_t max_blocks = std::max(((max_threads - 1) / block_size + 1),
                                          static_cast<int64_t>(1));
      const int64_t grid_size =
          std::min(max_blocks, (num + block_size - 1) / block_size);
      int quant_stride = 1;
      for (int i = x_num_col_dims + 1; i < in_dims.size(); i++) {
        quant_stride *= in_dims[i];
      }
      DequantizeTwoScale<T>
          <<<grid_size, block_size, 0, dev_ctx.stream()>>>(in_data,
                                                           scale_one,
                                                           scale_two,
                                                           max_range,
                                                           num,
                                                           n_scales,
                                                           quant_stride,
                                                           out_data);
    }
  }
};

template struct DequantizeFunctor<phi::GPUContext, float>;
template struct DequantizeFunctor<phi::GPUContext, double>;
template struct ChannelDequantizeFunctor<phi::GPUContext, float>;
template struct ChannelDequantizeFunctor<phi::GPUContext, double>;

}  // namespace operators
}  // namespace paddle
