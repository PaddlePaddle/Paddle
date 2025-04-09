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

#include "paddle/fluid/inference/tensorrt/plugin/hard_swish_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

nvinfer1::Dims HardSwishPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *in_dims, int nb_inputs) TRT_NOEXCEPT {
  assert(nb_inputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = in_dims[0];
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

template <typename T>
__device__ T kMax(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__device__ T kMin(T a, T b) {
  return a < b ? a : b;
}

template <typename T, unsigned TPB>
__global__ void hard_swish_kernel(float threshold,
                                  float scale,
                                  float offset,
                                  int n,
                                  const T *input,
                                  T *output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < n) {
    const T in = input[idx];
    output[idx] = in / scale * kMin<T>(kMax<T>(in + offset, 0), threshold);
  }
}

int HardSwishPlugin::enqueue(int batch_size,
                             const void *const *inputs,
#if IS_TRT_VERSION_LT(8000)
                             void **outputs,
                             void *,
                             cudaStream_t stream) {
#else
                             void *const *outputs,
                             void *,
                             cudaStream_t stream) TRT_NOEXCEPT {
#endif
  const auto &input_dims = this->getInputDims(0);
  int num = batch_size;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }
  float threshold = threshold_;
  float scale = scale_;
  float offset = offset_;

  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;

  const float *input = static_cast<const float *>(inputs[0]);
  float *output = static_cast<float *>(outputs[0]);
  hard_swish_kernel<float, block_size><<<grid_size, block_size, 0, stream>>>(
      threshold, scale, offset, num, input, output);

  return cudaGetLastError() != cudaSuccess;
}

#if IS_TRT_VERSION_GE(6000)

nvinfer1::DimsExprs HardSwishPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  return inputs[0];
}

int HardSwishPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  int num = 1;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }
  float threshold = threshold_;
  float scale = scale_;
  float offset = offset_;
  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;
  const float *input = static_cast<const float *>(inputs[0]);
  float *output = static_cast<float *>(outputs[0]);
  hard_swish_kernel<float, block_size><<<grid_size, block_size, 0, stream>>>(
      threshold, scale, offset, num, input, output);

  return cudaGetLastError() != cudaSuccess;
}

nvinfer1::DataType HardSwishPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    common::errors::InvalidArgument(
                        "The Elementwise Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  return input_types[0];
}

bool HardSwishPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument(
          "The input of swish plugin should not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));
  (in_out && pos < (nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    return (in.type == nvinfer1::DataType::kFLOAT) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}
#endif
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
