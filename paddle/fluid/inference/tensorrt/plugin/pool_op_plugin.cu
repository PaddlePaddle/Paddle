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

#include "paddle/fluid/inference/tensorrt/plugin/pool_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/operators/math/pooling.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

PoolPlugin* CreatePoolPluginDeserialize(const void* buffer, size_t length) {
  return new PoolPlugin(buffer, length);
}
REGISTER_TRT_PLUGIN("pool_plugin", CreatePoolPluginDeserialize);

nvinfer1::Dims PoolPlugin::getOutputDimensions(int index,
                                               const nvinfer1::Dims* inputDims,
                                               int nbInputs) {
  assert(nbInputs == 1);
  assert(index == 0);
  assert(inputDims[0].nbDims == 3);
  nvinfer1::Dims const& input_dims = inputDims[0];

  nvinfer1::Dims output_dims = input_dims;

  output_dims.d[1] = output_shape_[1];
  output_dims.d[2] = output_shape_[2];
  return output_dims;
}

int PoolPlugin::enqueue(int batchSize, const void* const* inputs,
                        void** outputs, void* workspace, cudaStream_t stream) {
  auto const& input_dims = this->getInputDims(0);
  int input_size = 0;
  float const* idata = reinterpret_cast<float const*>(inputs[0]);
  float** odatas = reinterpret_cast<float**>(outputs);

  std::vector<int> input_shape = input_shape_;
  std::vector<int> output_shape = output_shape_;
  input_shape.insert(input_shape.begin(), batchSize);
  output_shape.insert(output_shape.begin(), batchSize);

  if (pool_type_ == PoolType::max) {
    paddle::operators::math::MaxPool<float> pool_process;
    paddle::operators::math::Pool2dDirectCUDAFunctor<
        paddle::operators::math::MaxPool<float>, float>
        pool2d_forward;
    pool2d_forward(idata, input_shape, output_shape, ksize_, strides_,
                   paddings_, pool_process, true, adaptive_, odatas[0], stream);
  } else if (pool_type_ == PoolType::avg) {
    paddle::operators::math::AvgPool<float> pool_process;
    paddle::operators::math::Pool2dDirectCUDAFunctor<
        paddle::operators::math::AvgPool<float>, float>
        pool2d_forward;
    pool2d_forward(idata, input_shape, output_shape, ksize_, strides_,
                   paddings_, pool_process, true, adaptive_, odatas[0], stream);
  }

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
