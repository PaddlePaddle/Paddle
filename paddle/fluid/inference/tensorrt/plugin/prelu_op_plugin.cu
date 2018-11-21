// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>
#include <cassert>
#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/prelu_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

static const int CUDA_NUM_THREADS = 1024;
static const int CUDA_MAX_NUM_BLOCKS = 65535;
inline static int GET_NUM_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void PReluChannelWiseKernel(const float *input, const float *alpha,
                                       float *output, int channel,
                                       size_t spatial_size) {
  size_t offset = blockIdx.x * spatial_size;
  const float *in = input + offset;
  float *out = output + offset;
  float scale = alpha[blockIdx.x % channel];

  for (size_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
    float x = in[i];
    out[i] = (x > 0) ? x : scale * x;
  }
}

__global__ void PReluElementWiseKernel(const float *input, const float *alpha,
                                       float *output, size_t spatial_size) {
  size_t offset = blockIdx.x * spatial_size;
  const float *in = input + offset;
  const float *scale = alpha + offset;
  float *out = output + offset;

  for (size_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
    float x = in[i];
    out[i] = (x > 0) ? x : scale[i] * x;
  }
}

__global__ void PReluScalarKernel(const float *input, const float *alpha,
                                  float *output, size_t spatial_size) {
  size_t offset = blockIdx.x * spatial_size;
  const float *in = input + offset;
  float scale = *alpha;
  float *out = output + offset;

  for (size_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
    float x = in[i];
    out[i] = (x > 0) ? x : scale * x;
  }
}

static inline void PReluChannelWise(cudaStream_t stream, const float *input,
                                    const float *alpha, float *output,
                                    int batch_size,
                                    const nvinfer1::Dims &dims) {
  size_t unroll = batch_size * dims.d[0];
  size_t spatial_size = dims.d[1] * dims.d[2];
  CHECK_LT(unroll, CUDA_MAX_NUM_BLOCKS);
  PReluChannelWiseKernel<<<unroll, CUDA_NUM_THREADS, 0, stream>>>(
      input, alpha, output, dims.d[0], spatial_size);
}

static inline void PReluElementWise(cudaStream_t stream, const float *input,
                                    const float *alpha, float *output,
                                    int batch_size,
                                    const nvinfer1::Dims &dims) {
  size_t unroll = batch_size * dims.d[0];
  size_t spatial_size = dims.d[1] * dims.d[2];
  CHECK_LT(unroll, CUDA_MAX_NUM_BLOCKS);
  PReluElementWiseKernel<<<unroll, CUDA_NUM_THREADS, 0, stream>>>(
      input, alpha, output, spatial_size);
}

static inline void PReluScalar(cudaStream_t stream, const float *input,
                               const float *alpha, float *output,
                               int batch_size, const nvinfer1::Dims &dims) {
  size_t unroll = batch_size * dims.d[0];
  size_t spatial_size = dims.d[1] * dims.d[2];
  CHECK_LT(unroll, CUDA_MAX_NUM_BLOCKS);
  PReluScalarKernel<<<unroll, CUDA_NUM_THREADS, 0, stream>>>(
      input, alpha, output, spatial_size);
}

nvinfer1::Dims PReluPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

int PReluPlugin::enqueue(int batchSize, const void *const *inputs,
                         void **outputs, void *workspace, cudaStream_t stream) {
  // input dims is CHW.
  const auto &input_dims = this->getInputDims(0);
  const float *input = reinterpret_cast<const float *>(inputs[0]);
  const float *alpha = reinterpret_cast<const float *>(alpha_.get().values);
  float *output = reinterpret_cast<float **>(outputs)[0];
  if (mode_ == "channel") {
    PReluChannelWise(stream, input, alpha, output, batchSize, input_dims);
  } else if (mode_ == "element") {
    PReluElementWise(stream, input, alpha, output, batchSize, input_dims);
  } else {
    PReluScalar(stream, input, alpha, output, batchSize, input_dims);
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
