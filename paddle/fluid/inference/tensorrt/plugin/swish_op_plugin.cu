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
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/swish_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

SwishPlugin *CreateSwishPluginDeserialize(const void *buffer, size_t length) {
  return new SwishPlugin(buffer, length);
}
REGISTER_TRT_PLUGIN("swish_plugin", CreateSwishPluginDeserialize);

int SwishPlugin::initialize() { return 0; }

nvinfer1::Dims SwishPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}
__global__ void swish_kernel(int num, const float *input, float *output,
                             float beta) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
#if __CUDA_ARCH__ >= 350
    output[index] =
        __ldg(input + index) / (1.0f + expf(-beta * __ldg(input + index)));
#else
    output[index] = input[index] / (1.0f + expf(-beta * input[index]));
#endif
  }
}

int SwishPlugin::enqueue(int batch_size, const void *const *inputs,
                         void **outputs, void *workspace, cudaStream_t stream) {
  // input dims is CHW.
  const auto &input_dims = this->getInputDims(0);
  const float *input = reinterpret_cast<const float *>(inputs[0]);
  float *output = reinterpret_cast<float **>(outputs)[0];
  int num = batch_size;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }
  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  swish_kernel<<<blocks, threads, 0, stream>>>(num, input, output, beta_);

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
