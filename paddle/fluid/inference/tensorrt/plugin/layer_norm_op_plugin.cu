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
#include "paddle/fluid/inference/tensorrt/plugin/layer_norm_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/operators/layer_norm_op.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

LayerNormPlugin *CreateLayerNormPluginDeserialize(const void *buffer,
                                                  size_t length) {
  return new LayerNormPlugin(buffer, length);
}
REGISTER_TRT_PLUGIN("layer_norm_plugin", CreateLayerNormPluginDeserialize);

int LayerNormPlugin::initialize() {
  // cudaMalloc(&p_gpu_bias_, sizeof(float) * bias_.size());
  // cudaMemcpy(p_gpu_bias_, bias_.data(), bias_.size() * sizeof(float),
  //            cudaMemcpyHostToDevice);
  // cudaMalloc(&p_gpu_scale_, sizeof(float) * scale_.size());
  // cudaMemcpy(p_gpu_scale_, scale_.data(), scale_.size() * sizeof(float),
  //            cudaMemcpyHostToDevice);
  return 0;
}

nvinfer1::Dims LayerNormPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *inputDims, int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

int LayerNormPlugin::enqueue(int batch_size, const void *const *inputs,
                             void **outputs, void *workspace,
                             cudaStream_t stream) {
  const auto &input_dims = this->getInputDims(0);
  const float *input = reinterpret_cast<const float *>(inputs[0]);
  float *output = reinterpret_cast<float **>(outputs)[0];
  int begin_norm_axis = begin_norm_axis_;
  float eps = eps_;

  int n = batch_size;
  int c = input_dims.d[begin_norm_axis - 1];

  scale_t.Resize(framework::make_ddim({n, c}));
  bias_t.Resize(framework::make_ddim({n, c}));
  int device_id;
  cudaGetDevice(&device_id);
  float *scale_d = scale_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *bias_d = bias_t.mutable_data<float>(platform::CUDAPlace(device_id));

  for (int i = 0; i < n; i++) {
    cudaMemcpyAsync(scale_d + i * c, scale_.data(), sizeof(float) * c,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(bias_d + i * c, bias_.data(), sizeof(float) * c,
                    cudaMemcpyHostToDevice, stream);
  }

  // const float *bias = p_gpu_bias_;
  // const float *scale = p_gpu_scale_;

  std::vector<int> input_shape;
  input_shape.push_back(batch_size);
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape.push_back(input_dims.d[i]);
  }

  paddle::operators::LayerNormDirectCUDAFunctor<float> layer_norm;
  layer_norm(stream, input, input_shape, bias_d, scale_d, output, mean_shape_,
             variance_shape_, begin_norm_axis, eps);

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
