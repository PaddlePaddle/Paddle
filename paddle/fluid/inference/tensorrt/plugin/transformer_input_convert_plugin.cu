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

__global__ void TransformerInputConvertKernel(const int64_t* input,
                                              int32_t* output0) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int32_t shared_data;
  if (threadIdx.x == static_cast<int>(input[tid])) {
    atomicAdd(&shared_data, 1);
  }
  output0[0] = 0;
  output0[blockIdx.x + 1] = shared_data;
  __syncthreads();
  for (int i = 0; i < blockDim.x; ++i) {
    output0[i + 1] += output0[i];
  }
}

nvinfer1::DataType TransformerInputConvertPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return nvinfer1::DataType::kINT32;
}

nvinfer1::DimsExprs TransformerInputConvertPlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs output_dims{};
  output_dims.nbDims = 1;
  if (outputIndex == 0) {  // PosId
    const auto* one = exprBuilder.constant(1);
    output_dims.d[0] = exprBuilder.operation(
        nvinfer1::DimensionOperation::kSUM, *inputs[0].d[0], *one);
  } else {  // MaxSeqlen
    output_dims.d[0] = inputs[0].d[1];
  }
  return output_dims;
}

bool TransformerInputConvertPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nbInputs,
                    1,
                    platform::errors::InvalidArgument("Must have 1 inputs, "
                                                      "but got %d input(s). ",
                                                      nbInputs));
  PADDLE_ENFORCE_EQ(nbOutputs,
                    getNbOutputs(),
                    platform::errors::InvalidArgument("Must have 2 output, "
                                                      "but got %d output(s). ",
                                                      nbOutputs));
  if (pos == 0) {  // input
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  } else {  // output0, output1
    return inOut[pos].type == nvinfer1::DataType::kINT32 &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }
}

void TransformerInputConvertPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int nbOutputs) TRT_NOEXCEPT {}

void TransformerInputConvertPlugin::attachToContext(
    cudnnContext* cudnnContext,
    cublasContext* cublasContext,
    nvinfer1::IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {}

void TransformerInputConvertPlugin::detachFromContext() TRT_NOEXCEPT {}

void TransformerInputConvertPlugin::terminate() TRT_NOEXCEPT {}

int TransformerInputConvertPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  const auto input_desc = inputDesc[0];
  const int64_t* input = static_cast<const int64_t*>(inputs[0]);
  int32_t* output0 = static_cast<int32_t*>(outputs[0]);  // PosId
  // int32_t* output1 = static_cast<int32_t*>(outputs[1]);    // MaxSeqlen

  const int32_t num_blocks = input_desc.dims.d[0];   // batchs
  const int32_t num_threads = input_desc.dims.d[1];  // max sequnce length

  TransformerInputConvertKernel<<<num_blocks, num_threads, 0, stream>>>(
      input, output0);
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
