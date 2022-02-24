// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <cassert>
#include <cstring>
#include "paddle/fluid/inference/tensorrt/plugin/group_norm_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <unsigned TPB, typename T>
__global__ void scale_shift_kernel(const int n, const float* scale,
                                   const float* bias, T* output) {
  const float b = bias[blockIdx.y];
  const float s = scale[blockIdx.y];

  const int offset = (blockIdx.z * gridDim.y + blockIdx.y) * n;
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < n) {
    output[offset + idx] = s * output[offset + idx] + b;
  }
}

int GroupNormPlugin::enqueue(int batch_size, const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
                             void** outputs, void*, cudaStream_t stream) {
#else
                             void* const* outputs, void*,
                             cudaStream_t stream) TRT_NOEXCEPT {
#endif
  std::cout << "00000000" << std::endl;
  const int c = input_dims_[0];
  const int h = input_dims_[1];
  const int w = input_dims_[2];
  const int group_size = c / groups_;
  const int channel_volume = h * w;

  platform::dynload::cudnnSetTensor4dDescriptor(
      desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, batch_size * groups_,
      group_size, channel_volume);
  platform::dynload::cudnnDeriveBNTensorDescriptor(bn_desc_, desc_,
                                                   CUDNN_BATCHNORM_SPATIAL);
  platform::dynload::cudnnSetStream(handle_, stream);

  float alpha = 1.F;
  float beta = 0.F;
  const void* input = inputs[0];
  void* output = outputs[0];

  platform::dynload::cudnnBatchNormalizationForwardTraining(
      handle_, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, desc_, input, desc_,
      output, bn_desc_, bn_scale_, bn_bias_, 0.0, nullptr, nullptr, epsilon_,
      nullptr, nullptr);

  const int block_size = 256;
  const int grid_size = (channel_volume + block_size - 1) / block_size;
  const dim3 grid(grid_size, c, batch_size);
  std::cout << "7777777777" << std::endl;

  scale_shift_kernel<block_size, float><<<grid, block_size, 0, stream>>>(
      channel_volume, static_cast<const float*>(inputs[1]),
      static_cast<const float*>(inputs[2]), static_cast<float*>(output));

  // if (with_fp16_) {
  //   scale_shift_kernel<block_size, float16><<<grid, block_size, 0, stream>>>(
  //       channel_volume, static_cast<const float*>(inputs[1]),
  //       static_cast<const float*>(inputs[2]), static_cast<float16*>(output));
  // } else {
  //   scale_shift_kernel<block_size, float><<<grid, block_size, 0, stream>>>(
  //     channel_volume, static_cast<const float*>(inputs[1]), static_cast<const
  //     float*>(inputs[2]), static_cast<float*>(output));
  // }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
