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

#include <cassert>
#include <cstring>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/gelu_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// constants for approximating the normal cdf
constexpr float A = 0.5;

constexpr float B = 0.7978845608028654;  // sqrt(2.0/M_PI)

constexpr float C = 0.035677408136300125;  // 0.044715 * sqrt(2.0/M_PI)

GeluPlugin* CreateGeluPluginDeserialize(const void* buffer, size_t length) {
  return new GeluPlugin(buffer, length);
}
REGISTER_TRT_PLUGIN("gelu plugin", CreateGeluPluginDeserialize);

nvinfer1::Dims GeluPlugin::getOutputDimensions(int index,
                                               const nvinfer1::Dims* in_dims,
                                               int nb_inputs) {
  assert(nb_inputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const& input_dims = in_dims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

template <typename T, unsigned TPB>
__global__ void geluKernel(const T a, const T b, const T c, int n,
                           const T* input, T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < n) {
    const T in = input[idx];
    const T cdf = a + a * tanh(in * (c * in * in + b));
    output[idx] = in * cdf;
  }
}

int computeGelu(cudaStream_t stream, int n, const float* input, float* output) {
  constexpr int blockSize = 256;
  const int gridSize = (n + blockSize - 1) / blockSize;
  geluKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(
      A, B, C, n, input, output);

  CHECK(cudaPeekAtLastError());
  return 0;
}

int GeluPlugin::enqueue(int batchSize, const void* const* inputs,
                        void** outputs, void*, cudaStream_t stream) {
  int status = -1;

  const float* input = static_cast<const float*>(inputs[0]);
  float* output = static_cast<float*>(outputs[0]);
  status = computeGelu(stream, input_volume_ * batchSize, input, output);

  return status;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
