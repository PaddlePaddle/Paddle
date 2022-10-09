// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/plugin/roformer_op_plugin.h"

namespace plf = paddle::platform;
namespace dyl = paddle::platform::dynload;
namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {


template<typename T>
__global__ void RoformerKernel(
    const T *inputact,
    const T*input1,
    const T*intput2,
    T *output,
    const int nElement,
    const int N,
    const int H) {

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nElement)
      return;
  float qkv_index = (index % (3*N*H)) / static_cast<float>(N*H);
  if (qkv_index >= 2.0) {
      output[index] = inputact[index];
      return;
  }
  T _1 = input1[index] * inputact[index];
  int col = index % H;
  int half_lastdim = H/2;
  const int new_index = index - col + (col+half_lastdim) % H;
  output[index] = _1 + intput2[index] * inputact[new_index];
  return;
}


// class RoformerPlugin
RoformerPlugin::RoformerPlugin(
    const std::string &name,
    int head_num,
    int head_size):
    name_(name), head_num_(head_num), head_size_(head_size) {
  WHERE_AM_I();
}

RoformerPlugin::RoformerPlugin(
    const std::string &name,
    const void *buffer,
    size_t length):
    name_(name) {
  WHERE_AM_I();
  const char *data   = reinterpret_cast<const char *>(buffer);
  size_t offset = 0;
  memcpy(&head_num_, data + offset, sizeof(head_num_));
  offset += sizeof(head_num_);
  memcpy(&head_size_, data + offset, sizeof(head_size_));
}

RoformerPlugin::~RoformerPlugin() {
  WHERE_AM_I();
}

nvinfer1::IPluginV2DynamicExt *RoformerPlugin::clone() const noexcept {
  WHERE_AM_I();
  auto p = new RoformerPlugin(name_, head_num_, head_size_);
  p->setPluginNamespace(namespace_.c_str());
  return p;
}

int32_t RoformerPlugin::getNbOutputs() const noexcept {
  WHERE_AM_I();
  return 1;
}

nvinfer1::DataType RoformerPlugin::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const *inputTypes,
    int32_t nbInputs) const noexcept {
  WHERE_AM_I()
  assert(inputTypes[0] == nvinfer1::DataType::kFLOAT ||
         inputTypes[0] == nvinfer1::DataType::kHALF);
  return inputTypes[0];
}

nvinfer1::DimsExprs RoformerPlugin::getOutputDimensions(
    int32_t outputIndex,
    const nvinfer1::DimsExprs *inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  WHERE_AM_I()
  return  inputs[0];
}

bool RoformerPlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc *inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  WHERE_AM_I()
  if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
      return false;

  switch (pos) {  // 3 input and 1 output
  case 0:
      return inOut[0].type == nvinfer1::DataType::kFLOAT ||
             inOut[0].type == nvinfer1::DataType::kHALF;
  case 1:
      return inOut[1].type == inOut[0].type &&
             inOut[1].format == inOut[0].format;
  case 2:
      return inOut[2].type == inOut[0].type &&
             inOut[2].format == inOut[0].format;
  case 3:
      return inOut[3].type == inOut[0].type;
  default:  // should NOT be here!
      return false;
  }
  return false;
}

void RoformerPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int32_t nbOutputs) noexcept {
  WHERE_AM_I();
}

size_t RoformerPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int32_t nbOutputs) const noexcept {
  WHERE_AM_I()
  int tElement = 1;
  for (int i = 0; i < inputs[0].dims.nbDims; i++) {
    tElement *= inputs[0].dims.d[i];
  }
  return 0;
}

int32_t RoformerPlugin::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) noexcept {
  WHERE_AM_I()

  // input[0], (B, S, 3 *N*H)
  // varlen: (S,3*N*H,1,1)
  int nElement = 1;
  for (int i = 0; i < inputDesc[0].dims.nbDims; i++) {
      nElement *= inputDesc[0].dims.d[i];
  }

  int B = 1;
  int S = inputDesc[0].dims.d[0];
  int offset = 0;
  // dim3 grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1);
  // dim3 grid(CEIL_DIVIDE(nElement, 512*2), 1, 1), block(512, 1, 1);
  // dim3 grid(CEIL_DIVIDE(nElement, 512), 1, 1), block(512, 1, 1);

  int grid = CEIL_DIVIDE(nElement, 512);
  int block = 512;
  if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
    (RoformerKernel<float>)<<<grid, block, 0, stream>>>(
      reinterpret_cast<const float *>(inputs[0]),
      reinterpret_cast<const float *>(inputs[1]),
      reinterpret_cast<const float *>(inputs[2]),
      reinterpret_cast<float *>(outputs[0]),
      nElement,
      head_num_,
      head_size_);
  } else {  // DataType::kHALF
    (RoformerKernel<half>)<<<grid, block, 0, stream>>>(
      reinterpret_cast<const half *>(inputs[0]),
      reinterpret_cast<const half *>(inputs[1]),
      reinterpret_cast<const half *>(inputs[2]),
      reinterpret_cast<half *>(outputs[0]),
      nElement,
      head_num_,
      head_size_);
  }
  return 0;
}

int32_t RoformerPlugin::initialize() noexcept {
    WHERE_AM_I();
    return 0;
}

void RoformerPlugin::terminate() noexcept {
    WHERE_AM_I();
}

void RoformerPlugin::destroy() noexcept {
    WHERE_AM_I();
    delete this;
    return;
}

size_t RoformerPlugin::getSerializationSize() const noexcept {
    WHERE_AM_I();
    return sizeof(head_num_) + sizeof(head_size_);
}

void RoformerPlugin::serialize(void *buffer) const noexcept {
    WHERE_AM_I()
    size_t offset = 0;
    char * data   = reinterpret_cast<char *>(buffer);
    memcpy(data + offset, &head_num_, sizeof(head_num_));
    offset += sizeof(head_num_);
    memcpy(data + offset, &head_size_, sizeof(head_size_));
}

void RoformerPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
    WHERE_AM_I()
    namespace_ = pluginNamespace;
}
const char *RoformerPlugin::getPluginNamespace() const noexcept {
    WHERE_AM_I()
    return namespace_.c_str();
}

const char *RoformerPlugin::getPluginType() const noexcept {
    WHERE_AM_I()
    return "roformerplugin";
}

const char *RoformerPlugin::getPluginVersion() const noexcept {
    WHERE_AM_I()
    return "1";
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
