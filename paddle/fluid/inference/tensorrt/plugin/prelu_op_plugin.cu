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
#include "paddle/fluid/inference/tensorrt/plugin/prelu_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/operators/math/prelu.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

PReluPlugin *CreatePreluPluginDeserialize(const void *buffer, size_t length) {
  return new PReluPlugin(buffer, length);
}
REGISTER_TRT_PLUGIN("prelu_plugin", CreatePreluPluginDeserialize);

int PReluPlugin::initialize() {
  cudaMalloc(&p_gpu_weight_, sizeof(float) * weight_.size());
  cudaMemcpy(p_gpu_weight_, weight_.data(), weight_.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  return 0;
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

int PReluPlugin::enqueue(int batch_size, const void *const *inputs,
                         void **outputs, void *workspace, cudaStream_t stream) {
  // input dims is CHW.
  const auto &input_dims = this->getInputDims(0);
  const float *input = reinterpret_cast<const float *>(inputs[0]);
  // const float *alpha = reinterpret_cast<const float *>(alpha_.get().values);
  const float *alpha = p_gpu_weight_;
  float *output = reinterpret_cast<float **>(outputs)[0];

  std::vector<int> input_shape;
  input_shape.push_back(batch_size);
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape.push_back(input_dims.d[i]);
  }

  if (mode_ == "channel") {
    operators::math::PreluChannelWiseDirectCUDAFunctor<float>
        prelu_channel_wise;
    prelu_channel_wise(stream, input, alpha, output, input_shape);
  } else if (mode_ == "element") {
    operators::math::PreluElementWiseDirectCUDAFunctor<float>
        prelu_element_wise;
    prelu_element_wise(stream, input, alpha, output, input_shape);
  } else {
    operators::math::PreluScalarDirectCUDAFunctor<float> prelu_scalar;
    prelu_scalar(stream, input, alpha, output, input_shape);
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
