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

template <typename T, unsigned TPB>
__global__ void group_norm_kernel(const int n, const float *scale,
                                  const float *bias, T *output) {
  const T b = bias[blockIdx.y];
  const T s = scale[blockIdx.y];

  const int offset = (blockIdx.z * gridDim.y + blockIdx.y) * n;
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < n) {
    output[offset + idx] = s * output[offset + idx] + b;
  }
}

int GroupNormPlugin::enqueue(int batch_size, const void *const *inputs,
#if IS_TRT_VERSION_LT(8000)
                             void **outputs, void *, cudaStream_t stream) {
#else
                             void *const *outputs, void *,
                             cudaStream_t stream) TRT_NOEXCEPT {
#endif
  std::cout << "00000000" << std::endl;
  const int c = input_dims_[0];
  const int h = input_dims_[1];
  const int w = input_dims_[2];
  const int group_size = c / groups_;
  const int channel_volume = h * w;
  std::cout << "111111111" << std::endl;

  int max_batch_size = batch_size + 3;
  cudaMalloc(&bn_scale_, max_batch_size * c * sizeof(float));
  cudaMalloc(&bn_bias_, max_batch_size * c * sizeof(float));

  std::vector<float> ones(c, 1.F);
  std::vector<float> zeroes(c, 0.F);
  for (int i = 0; i < max_batch_size; i++) {
    cudaMemcpy(bn_scale_ + i * c, ones.data(), sizeof(float) * c,
               cudaMemcpyHostToDevice);
    cudaMemcpy(bn_bias_ + i * c, zeroes.data(), sizeof(float) * c,
               cudaMemcpyHostToDevice);
  }

  std::cout << "3333333333" << std::endl;
  platform::dynload::cudnnSetTensor4dDescriptor(
      desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, batch_size * groups_,
      group_size, channel_volume);
  platform::dynload::cudnnDeriveBNTensorDescriptor(bn_desc_, desc_,
                                                   CUDNN_BATCHNORM_SPATIAL);
  platform::dynload::cudnnSetStream(handle_, stream);
  std::cout << "4444444444" << std::endl;
  float alpha = 1.F;
  float beta = 0.F;
  const float *input = static_cast<const float *>(inputs[0]);
  float *output = static_cast<float *>(outputs[0]);
  std::cout << "55555555" << std::endl;
  platform::dynload::cudnnBatchNormalizationForwardTraining(
      handle_, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, desc_, input, desc_,
      output, bn_desc_, bn_scale_, bn_bias_, 0.0, nullptr, nullptr, epsilon_,
      nullptr, nullptr);
  std::cout << "666666666" << std::endl;
  const int block_size = 256;
  const int grid_size = (channel_volume + block_size - 1) / block_size;
  const dim3 grid(grid_size, c, batch_size);
  std::cout << "7777777777" << std::endl;
  group_norm_kernel<float, block_size><<<grid, block_size, 0, stream>>>(
      channel_volume, static_cast<const float *>(inputs[1]),
      static_cast<const float *>(inputs[2]), output);
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
