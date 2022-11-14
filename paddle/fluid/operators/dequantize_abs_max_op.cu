/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/dequantize_abs_max_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void KeDequantize(
    const T* in, const float* scale, float max_range, int num, float* out) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num) {
    out[idx] = in[idx] * scale[0] / max_range;
  }
}

template <typename T>
struct DequantizeFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor* scale,
                  float max_range,
                  phi::DenseTensor* out) {
    const T* in_data = in->data<T>();
    const float* scale_factor = scale->data<float>();
    float* out_data = out->mutable_data<float>(dev_ctx.GetPlace());

    int num = in->numel();
    int block = 512;
    int grid = (num + block - 1) / block;

    KeDequantize<T><<<grid, block, 0, dev_ctx.stream()>>>(
        in_data, scale_factor, max_range, num, out_data);
  }
};

template struct DequantizeFunctor<phi::GPUContext, int8_t>;
template struct DequantizeFunctor<phi::GPUContext, int16_t>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = phi::GPUContext;
REGISTER_OP_CUDA_KERNEL(dequantize_abs_max,
                        ops::DequantizeMaxAbsKernel<CUDA, int8_t>,
                        ops::DequantizeMaxAbsKernel<CUDA, int16_t>);
