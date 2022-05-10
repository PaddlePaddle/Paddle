/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/plugin/transformer_input_convert_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

nvinfer1::DataType TransformerInputConvertPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

nvinfer1::DimsExprs TransformerInputConvertPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs output_dims(inputs[0]);
  if (transB_) {
    output_dims.d[output_dims.nbDims - 2] = inputs[0].d[inputs[0].nbDims - 1];
  } else {
    output_dims.d[output_dims.nbDims - 2] = inputs[0].d[inputs[0].nbDims - 2];
  }
  if (transA_) {
    output_dims.d[output_dims.nbDims - 1] = inputs[1].d[inputs[1].nbDims - 2];
  } else {
    output_dims.d[output_dims.nbDims - 1] = inputs[1].d[inputs[1].nbDims - 1];
  }
  return output_dims;
}

bool TransformerInputConvertPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
    int nbOutputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nbInputs, 2,
                    platform::errors::InvalidArgument("Must have 2 inputs, "
                                                      "but got %d input(s). ",
                                                      nbInputs));
  PADDLE_ENFORCE_EQ(nbOutputs, getNbOutputs(),
                    platform::errors::InvalidArgument("Must have 1 output, "
                                                      "but got %d output(s). ",
                                                      nbOutputs));
  if (pos == 0) {
    return (inOut[pos].type == nvinfer1::DataType::kHALF ||
            inOut[pos].type == nvinfer1::DataType::kFLOAT ||
            inOut[pos].type == nvinfer1::DataType::kINT8) &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }
}

void TransformerInputConvertPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int nbOutputs) TRT_NOEXCEPT {}

void TransformerInputConvertPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext,
    nvinfer1::IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {}

void TransformerInputConvertPlugin::detachFromContext() TRT_NOEXCEPT {}

// When tensorrt engine freed ,there is "double free" ERROR. TODO@Wangzheee
void TransformerInputConvertPlugin::terminate() TRT_NOEXCEPT {}

int TransformerInputConvertPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT {
  const auto Input0Desc = inputDesc[0];
  const auto Input1Desc = inputDesc[1];
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
