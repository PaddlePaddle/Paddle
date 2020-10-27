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
#include "paddle/fluid/inference/tensorrt/plugin/special_slice_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)
SpecialSlicePluginDynamic::SpecialSlicePluginDynamic() {}

SpecialSlicePluginDynamic::SpecialSlicePluginDynamic(void const* serial_data,
                                                     size_t serial_length) {}

SpecialSlicePluginDynamic::~SpecialSlicePluginDynamic() {}

nvinfer1::IPluginV2DynamicExt* SpecialSlicePluginDynamic::clone() const {
  return new SpecialSlicePluginDynamic();
}

const char* SpecialSlicePluginDynamic::getPluginType() const {
  return "special_slice_plugin";
}

int SpecialSlicePluginDynamic::getNbOutputs() const { return 1; }

int SpecialSlicePluginDynamic::initialize() { return 0; }

size_t SpecialSlicePluginDynamic::getSerializationSize() const {
  size_t serialize_size = 0;
  return serialize_size;
}

void SpecialSlicePluginDynamic::serialize(void* buffer) const {}

nvinfer1::DimsExprs SpecialSlicePluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) {
  nvinfer1::DimsExprs output(inputs[0]);
  auto one = expr_builder.constant(1);
  output.d[0] = expr_builder.operation(nvinfer1::DimensionOperation::kSUB,
                                       *inputs[1].d[0], *one);

  return output;
}

void SpecialSlicePluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {}

size_t SpecialSlicePluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
  return 0;
}

void SpecialSlicePluginDynamic::destroy() { delete this; }

void SpecialSlicePluginDynamic::terminate() {}

bool SpecialSlicePluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* desc, int nb_inputs,
    int nb_outputs) {
  if (pos == 0)  // slice tensor
    return (desc[pos].type == nvinfer1::DataType::kHALF &&
            desc[pos].format ==
                nvinfer1::TensorFormat::kLINEAR);  // || desc[pos].type ==
  // nvinfer1::DataType::kFLOAT);

  if (pos == 1)  // cu_seqlen
    return (desc[pos].type == nvinfer1::DataType::kINT32 &&
            desc[pos].format == nvinfer1::TensorFormat::kLINEAR);

  return (desc[pos].type == nvinfer1::DataType::kHALF &&
          desc[pos].format ==
              nvinfer1::TensorFormat::kLINEAR);  // || desc[pos].type ==
  // nvinfer1::DataType::kFLOAT);
}

nvinfer1::DataType SpecialSlicePluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0, platform::errors::InvalidArgument(
                                  "The index should be equal to 0"));
  return input_types[0];
}

template <typename T>
__global__ void SpecialSliceKernel(const T* slice_input,
                                   const int32_t* cu_seqlens, T* output) {
  const int hidden = blockDim.x;
  const int batch = blockIdx.x;

  output[batch * hidden + threadIdx.x] =
      slice_input[cu_seqlens[batch] * hidden + threadIdx.x];
}

int SpecialSlicePluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) {
  auto input_dims = input_desc[0].dims;  // (sum(S), 768, 1, 1)
  auto out_dims = output_desc[0].dims;   // (batch, 768, 1, 1)

  assert(input_desc[0].type == nvinfer1::DataType::kHALF);

  const int32_t hidden = input_dims.d[1];
  const int num_blocks = out_dims.d[0];  // batch size
  const int num_threads = hidden;

  const half* slice_input = static_cast<const half*>(inputs[0]);
  const int32_t* cu_seqlens = static_cast<const int32_t*>(inputs[1]);
  half* output = static_cast<half*>(outputs[0]);

  SpecialSliceKernel<<<num_blocks, num_threads, 0, stream>>>(
      slice_input, cu_seqlens, output);

  return cudaGetLastError() != cudaSuccess;
}

SpecialSlicePluginDynamicCreator::SpecialSlicePluginDynamicCreator() {}

const char* SpecialSlicePluginDynamicCreator::getPluginName() const {
  return "special_slice_plugin";
}

const char* SpecialSlicePluginDynamicCreator::getPluginVersion() const {
  return "1";
}

const nvinfer1::PluginFieldCollection*
SpecialSlicePluginDynamicCreator::getFieldNames() {
  return &field_collection_;
}

nvinfer1::IPluginV2* SpecialSlicePluginDynamicCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  return new SpecialSlicePluginDynamic();
}

nvinfer1::IPluginV2* SpecialSlicePluginDynamicCreator::deserializePlugin(
    const char* name, const void* serial_data, size_t serial_length) {
  auto plugin = new SpecialSlicePluginDynamic(serial_data, serial_length);
  return plugin;
}

void SpecialSlicePluginDynamicCreator::setPluginNamespace(
    const char* lib_namespace) {
  plugin_namespace_ = lib_namespace;
}

const char* SpecialSlicePluginDynamicCreator::getPluginNamespace() const {
  return plugin_namespace_.c_str();
}

#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
