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
#include "paddle/fluid/inference/tensorrt/plugin/cast_int_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

nvinfer1::DimsExprs CastIntPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) {
  assert(output_index == 0);
  return inputs[0];
}

bool CastIntPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
    int nb_outputs) {
  const nvinfer1::PluginTensorDesc& in = in_out[pos];
  return (in.type == nvinfer1::DataType::kINT32);
}

nvinfer1::DataType CastIntPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0, platform::errors::InvalidArgument(
                                  "The Cast Int only has one input, so the "
                                  "index value should be 0, but get %d.",
                                  index));
  return input_types[index];
}

__global__ void castIntKernel(const int64_t* input, int32_t* output,
                              size_t num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) return;
  output[idx] = input[idx] + 1;
}

int CastIntPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                                  const nvinfer1::PluginTensorDesc* output_desc,
                                  const void* const* inputs,
                                  void* const* outputs, void* workspace,
                                  cudaStream_t stream) {
  auto input_dims = input_desc[0].dims;
  auto output_dims = output_desc[0].dims;
  size_t num_elements = ProductDim(input_dims);
  size_t out_num_elements = ProductDim(output_dims);

  assert(input_type ==
         nvinfer1::DataType::kINT32);  // although the input is int64_t
  assert(num_elements == out_num_elements);

  const size_t num_threads = 256;
  castIntKernel<<<num_elements / num_threads + 1, num_threads>>>(
      static_cast<const int64_t*>(inputs[0]), static_cast<int32_t*>(outputs[0]),
      num_elements);

  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
