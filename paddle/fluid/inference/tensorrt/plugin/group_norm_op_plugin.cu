// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_fp16.h>
#include <cstring>
#include "paddle/fluid/inference/tensorrt/plugin/group_norm_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <unsigned TPB, typename T>
__global__ void scale_shift_kernel(const int n, const float* scale,
                                   const float* bias, T* output);

template <>
__global__ void scale_shift_kernel<256, float>(const int n, const float* scale,
                                               const float* bias,
                                               float* output) {
  const float b = bias[blockIdx.y];
  const float s = scale[blockIdx.y];

  const int offset = (blockIdx.z * gridDim.y + blockIdx.y) * n;
  const int idx = blockIdx.x * 256 + threadIdx.x;

  if (idx < n) {
    output[offset + idx] = s * output[offset + idx] + b;
  }
}

template <>
__global__ void scale_shift_kernel<256, __half>(const int n, const float* scale,
                                                const float* bias,
                                                __half* output) {
  const float b = bias[blockIdx.y];
  const float s = scale[blockIdx.y];

  const int offset = (blockIdx.z * gridDim.y + blockIdx.y) * n;
  const int idx = blockIdx.x * 256 + threadIdx.x;

  if (idx < n) {
    output[offset + idx] =
        __hfma(__float2half(s), output[offset + idx], __float2half(b));
  }
}

int GroupNormPlugin::enqueue(int batch_size, const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
                             void** outputs, void*, cudaStream_t stream) {
#else
                             void* const* outputs, void*,
                             cudaStream_t stream) TRT_NOEXCEPT {
#endif
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

  int ans = 1;
  if (with_fp16_) {
    ans = enqueueImpl<__half>(inputs, outputs, handle_, desc_, bn_desc_,
                              bn_scale_, bn_bias_, epsilon_, channel_volume,
                              batch_size, c, stream);
  } else {
    ans = enqueueImpl<float>(inputs, outputs, handle_, desc_, bn_desc_,
                             bn_scale_, bn_bias_, epsilon_, channel_volume,
                             batch_size, c, stream);
  }
  return ans;
}

template <typename T>
#if IS_TRT_VERSION_LT(8000)
int GroupNormPlugin::enqueueImpl(const void* const* inputs, void** outputs,
#else
int GroupNormPlugin::enqueueImpl(const void* const* inputs,
                                 void* const* outputs,
#endif
                                 const cudnnHandle_t& handle,
                                 const cudnnTensorDescriptor_t& desc,
                                 const cudnnTensorDescriptor_t& bn_desc,
                                 float* bn_scale, float* bn_bias, float eps,
                                 const int channel_volume, const int batch_size,
                                 const int c, const cudaStream_t& stream) {
  float alpha = 1.F;
  float beta = 0.F;
  const T* input = static_cast<const T*>(inputs[0]);
  T* output = static_cast<T*>(outputs[0]);
  platform::dynload::cudnnBatchNormalizationForwardTraining(
      handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, desc, input, desc, output,
      bn_desc, bn_scale, bn_bias, 0.0, nullptr, nullptr, eps, nullptr, nullptr);

  const int block_size = 256;
  const int grid_size = (channel_volume + block_size - 1) / block_size;
  const dim3 grid(grid_size, c, batch_size);

  scale_shift_kernel<block_size, T><<<grid, block_size, 0, stream>>>(
      channel_volume, scale_, bias_, output);

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
