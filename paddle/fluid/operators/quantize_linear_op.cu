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

#include <string>

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/fake_quantize_op.cu.h"
#include "paddle/fluid/operators/quantize_linear_op.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

using float16 = paddle::platform::float16;

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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = phi::GPUContext;
REGISTER_OP_CUDA_KERNEL(dequantize_linear,
                        ops::DeQuantizeLinearKernel<CUDA, float>,
                        ops::DeQuantizeLinearKernel<CUDA, float16>,
                        ops::DeQuantizeLinearKernel<CUDA, int8_t>,
                        ops::DeQuantizeLinearKernel<CUDA, double>);

REGISTER_OP_CUDA_KERNEL(quantize_linear,
                        ops::QuantizeLinearKernel<CUDA, float>,
                        ops::QuantizeLinearKernel<CUDA, float16>);
