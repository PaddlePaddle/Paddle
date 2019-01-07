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
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/tensorrt/plugin/layer_norm_op_plugin.h"
#include "paddle/fluid/operators/math/layer_norm.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

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
  // input dims is CHW.
  const auto &input_dims = this->getInputDims(0);
  const float *input = reinterpret_cast<const float *>(inputs[0]);
  const float *scale = reinterpret_cast<const float *>(scale_.get().values);
  const float *bias = reinterpret_cast<const float *>(bias_.get().values);
  float *output = reinterpret_cast<float **>(outputs)[0];

  int left_num = 1;
  int right_num = 1;
  for (int i = 0; i < begin_norm_axis_; i++) {
    left_num *= input_dims.d[i];
  }

  for (int i = begin_norm_axis_; i < input_dims.nbDims; i++) {
    right_num *= input_dims.d[i];
  }
  platform::CUDAPlace place;
  framework::LoDTensor mean_tensor;
  framework::LoDTensor var_tensor;

  mean_tensor.Resize(framework::make_ddim({left_num * batch_size}));
  var_tensor.Resize(framework::make_ddim({left_num * batch_size}));
  float *mean_data = mean_tensor.mutable_data<float>(place);
  float *var_data = var_tensor.mutable_data<float>(place);

  paddle::operators::math::LayerNormDirectCUDAFunctor<float> layer_func;

  layer_func(stream, input, scale, bias, output, mean_data, var_data, epsilon_,
             right_num, left_num * batch_size);

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
