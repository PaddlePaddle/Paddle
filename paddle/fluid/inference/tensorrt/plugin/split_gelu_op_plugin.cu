// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "paddle/fluid/inference/tensorrt/plugin/split_gelu_op_plugin.h"
#include <vector>

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <typename T, int32_t HHS, int32_t TPB>
__global__ void splitGeLUKernel(T const *input,
                                T *output,
                                float const fDiv,
                                float const fAdd,
                                float const fMul) {
  int32_t indexInput = blockIdx.x * HHS * 2 + threadIdx.x;
  int32_t indexOutput = blockIdx.x * HHS + threadIdx.x;

#pragma unroll
  for (int32_t i = 0; i < HHS / TPB; ++i) {
    auto valueL = static_cast<float>(input[indexInput]);
    auto valueR = static_cast<float>(input[indexInput + HHS]);
    float tmp = valueR;
    tmp /= fDiv;
    tmp = erff(tmp);
    tmp += fAdd;
    tmp *= valueR;
    tmp *= fMul;
    tmp *= valueL;
    output[indexOutput] = static_cast<T>(tmp);
    indexInput += TPB;
    indexOutput += TPB;
  }
}

template <typename T>
int32_t launchSplitGeLUKernel(cudaStream_t stream,
                              int32_t gridSize,
                              int32_t nHalfHiddenSize,
                              T const *input,
                              T *output,
                              float const fDiv,
                              float const fAdd,
                              float const fMul) {
  constexpr int32_t TPB = 256;  // thread per block
  switch (nHalfHiddenSize) {
    case 1280:
      (splitGeLUKernel<T, 1280, TPB>)<<<gridSize, TPB, 0, stream>>>(
          input, output, fDiv, fAdd, fMul);
      break;
    case 2560:
      (splitGeLUKernel<T, 2560, TPB>)<<<gridSize, TPB, 0, stream>>>(
          input, output, fDiv, fAdd, fMul);
      break;
    case 5120:
      (splitGeLUKernel<T, 5120, TPB>)<<<gridSize, TPB, 0, stream>>>(
          input, output, fDiv, fAdd, fMul);
      break;
    default:
      PADDLE_THROW(
          platform::errors::Fatal("The function launchSplitGeLUKernel of "
                                  "SplitGeluPluginDynamic TRT Plugin "
                                  "encounter error"));
      break;
  }
  return 0;
}

void SplitGeluPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {}
bool SplitGeluPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
      return in.type == nvinfer1::DataType::kHALF &&
             in.format == nvinfer1::TensorFormat::kLINEAR;
    } else {
      return in.type == nvinfer1::DataType::kFLOAT &&
             in.format == nvinfer1::TensorFormat::kLINEAR;
    }
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType SplitGeluPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

nvinfer1::DimsExprs SplitGeluPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs output = inputs[0];
  output.d[2] = expr_builder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV,
                                       *inputs[0].d[2],
                                       *expr_builder.constant(2));
  return output;
}
int SplitGeluPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  const auto &input_dims = input_desc[0].dims;
  auto input_type = input_desc[0].type;
  const int32_t gridSize = input_desc[0].dims.d[0] * input_desc[0].dims.d[1];
  const int32_t nHalfHiddenSize = input_desc[0].dims.d[2] / 2;  // HHS
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(3) << "TRT Plugin DataType selected. SplitGelu-->fp32";
    launchSplitGeLUKernel(stream,
                          gridSize,
                          nHalfHiddenSize,
                          static_cast<float const *>(inputs[0]),
                          static_cast<float *>(outputs[0]),
                          1.4140625f,
                          1,
                          0.5f);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(3) << "TRT Plugin DataType selected. SplitGelu-->fp16";
    launchSplitGeLUKernel(stream,
                          gridSize,
                          nHalfHiddenSize,
                          static_cast<half const *>(inputs[0]),
                          static_cast<half *>(outputs[0]),
                          1.4140625f,
                          1,
                          0.5f);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The SplitGelu TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
