/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/fake_dequantize_functor.h"

namespace phi {
namespace funcs {

template <typename T>
__global__ void KeDequantize(
    const T* in, const T* scale, T max_range, int64_t num, T* out) {
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int64_t i = idx; i < num; i += blockDim.x * gridDim.x) {
    out[i] = in[i] * scale[0] / max_range;
  }
}

template <typename Context, typename T>
void DequantizeFunctor<Context, T>::operator()(const Context& dev_ctx,
                                               const DenseTensor* in,
                                               const DenseTensor* scale,
                                               T max_range,
                                               DenseTensor* out) {
  const T* in_data = in->data<T>();
  const T* scale_factor = scale->data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);

  int64_t num = in->numel();
  int64_t block_size =
      std::min(num, static_cast<int64_t>(dev_ctx.GetMaxThreadsPerBlock() / 4));
  int64_t max_threads =
      dev_ctx.GetMaxPhysicalThreadCount();  // SM * block_per_SM
  const int64_t max_blocks =
      std::max(((max_threads - 1) / block_size + 1), static_cast<int64_t>(1));
  const int64_t grid_size =
      std::min(max_blocks, (num + block_size - 1) / block_size);
  KeDequantize<T><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      in_data, scale_factor, max_range, num, out_data);
}

template class DequantizeFunctor<GPUContext, float>;
template class DequantizeFunctor<GPUContext, double>;
template class DequantizeFunctor<GPUContext, float16>;

}  // namespace funcs
}  // namespace phi
