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

#include "paddle/fluid/inference/tensorrt/plugin/recover_padding_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

__global__ void RecoverPaddingKernel(const float* input0,
                                     const int32_t* input1,
                                     float* output) {
  int word_id = blockIdx.x * gridDim.y + blockIdx.y;
  int32_t seqence_length = input1[blockIdx.x + 1] - input1[blockIdx.x];
  if (blockIdx.y < seqence_length) {
    output[word_id * gridDim.z * blockDim.x + blockIdx.z * blockDim.x +
           threadIdx.x] =
        input0[(input1[blockIdx.x] + blockIdx.y) * gridDim.z * blockDim.x +
               blockIdx.z * blockDim.x + threadIdx.x];
  } else {
    output[word_id * gridDim.z * blockDim.x + blockIdx.z * blockDim.x +
           threadIdx.x] = 0;
  }
}

nvinfer1::DataType RecoverPaddingPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

nvinfer1::DimsExprs RecoverPaddingPlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs output_dims{};
  output_dims.nbDims = 3;
  const auto* one = exprBuilder.constant(1);
  output_dims.d[0] = exprBuilder.operation(
      nvinfer1::DimensionOperation::kSUB, *inputs[1].d[0], *one);
  output_dims.d[1] = inputs[2].d[1];
  output_dims.d[2] = inputs[0].d[1];
  return output_dims;
}

bool RecoverPaddingPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nbInputs,
                    3,
                    platform::errors::InvalidArgument("Must have 3 inputs, "
                                                      "but got %d input(s). ",
                                                      nbInputs));
  PADDLE_ENFORCE_EQ(nbOutputs,
                    getNbOutputs(),
                    platform::errors::InvalidArgument("Must have 1 output, "
                                                      "but got %d output(s). ",
                                                      nbOutputs));
  if (pos == 1) {  // PosId
    return inOut[pos].type == nvinfer1::DataType::kINT32 &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  } else if (pos == 2) {  // mask_id
    return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  } else {
    return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }
  // return (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format
  // == nvinfer1::TensorFormat::kLINEAR)||
  // (inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format ==
  // nvinfer1::TensorFormat::kLINEAR)||
  // (inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format ==
  // nvinfer1::TensorFormat::kCHW32);
}

void RecoverPaddingPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int nbOutputs) TRT_NOEXCEPT {}

void RecoverPaddingPlugin::attachToContext(
    cudnnContext* cudnnContext,
    cublasContext* cublasContext,
    nvinfer1::IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {}

void RecoverPaddingPlugin::detachFromContext() TRT_NOEXCEPT {}

void RecoverPaddingPlugin::terminate() TRT_NOEXCEPT {}

int RecoverPaddingPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                  const nvinfer1::PluginTensorDesc* outputDesc,
                                  const void* const* inputs,
                                  void* const* outputs,
                                  void* workspace,
                                  cudaStream_t stream) TRT_NOEXCEPT {
  const auto input0_desc = inputDesc[0];
  const auto input1_desc = inputDesc[1];
  const auto input2_desc = inputDesc[2];
  const float* input0 = static_cast<const float*>(inputs[0]);
  const int32_t* input1 =
      static_cast<const int32_t*>(inputs[1]);  // pos_id_tensor
  float* output = static_cast<float*>(outputs[0]);
  int32_t num_threads;
  if (input0_desc.dims.d[1] % 512 == 0) {
    num_threads = 512;
  } else if (input0_desc.dims.d[1] % 256 == 0) {
    num_threads = 256;
  } else if (input0_desc.dims.d[1] % 128 == 0) {
    num_threads = 128;
  } else if (input0_desc.dims.d[1] % 64 == 0) {
    num_threads = 64;
  } else if (input0_desc.dims.d[1] % 32 == 0) {
    num_threads = 32;
  } else if (input0_desc.dims.d[1] % 16 == 0) {
    num_threads = 16;
  } else if (input0_desc.dims.d[1] % 8 == 0) {
    num_threads = 8;
  } else if (input0_desc.dims.d[1] % 4 == 0) {
    num_threads = 4;
  } else if (input0_desc.dims.d[1] % 2 == 0) {
    num_threads = 2;
  } else {
    num_threads = 1;
  }
  const dim3 num_blocks(
      input1_desc.dims.d[0] - 1,
      input2_desc.dims.d[1],
      input0_desc.dims.d[1] / num_threads);  //  batchs, max sequnce length
                                             //  (mask_id.dims.d[1]),
                                             //  input.dims.d[1]/256
  RecoverPaddingKernel<<<num_blocks, num_threads, 0, stream>>>(
      input0, input1, output);
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
