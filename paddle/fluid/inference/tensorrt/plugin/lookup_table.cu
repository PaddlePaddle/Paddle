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

#pragma once
#include "paddle/fluid/inference/tensorrt/plugin/lookup_table.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

char const* PLUGINVERSION{"1"};
char const* LOOKUPTABLEPLUGINNAME{"LookupTablePluginDynamic"};

template <typename T, unsigned TPB>
__global__ void lookup_table_kernel(int weight_height,
                                    int32_t const* inputIds,
                                    T const* wordEmb,
                                    int32_t const wordSize,
                                    T* output) {
  // 1. lookup word and token of the block
  // blockIdx.x = position in the sequence
  // blockIdx.y = batch
  // gridDim.x = S
  // gridDim.y = B
  __shared__ int wordId;
  int32_t const seqPos = blockIdx.x + blockIdx.y * gridDim.x;
  if (threadIdx.x == 0) {
    wordId = inputIds[seqPos];
  }
  __syncthreads();

  // 2. load word embeddings and add them toghether
  // offset into embeddings is given by wordId * hidden_size
  int32_t const woffset = wordId * weight_height;
  // the output offset is given by b * (S*hidden_size) + s * hidden_size
  int32_t const outOffset = seqPos * weight_height;
  if (wordId >= 0 && wordId < wordSize) {
    for (int it = threadIdx.x; it < weight_height; it += TPB) {
      T const w(wordEmb[woffset + it]);
      output[outOffset + it] = w;
    }
  } else {
    printf(
        "Error!!!!!!(LookupTablePlugin): ID cannot be lookup "
        "table: ID < 0 or ID > max ");
    return;
  }
}

template <typename T>
int lookup_table(cudaStream_t stream,
                 int weight_height,
                 int B,
                 int S,
                 int32_t const* inputIds,
                 T const* wordEmb,
                 int32_t const wordSize,
                 T* output) {
  constexpr int tpb = 256;
  dim3 const grid(S, B, 1);
  dim3 const block(tpb, 1, 1);
  lookup_table_kernel<T, tpb><<<grid, block, 0, stream>>>(
      weight_height, inputIds, wordEmb, wordSize, output);
  return 0;
}

// Static class fields initialization
nvinfer1::PluginFieldCollection LookupTablePluginDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField>
    LookupTablePluginDynamicCreator::mPluginAttributes;

LookupTablePluginDynamic::LookupTablePluginDynamic(
    nvinfer1::DataType const type,
    void* weight_dev,
    int32_t weight_size,
    int32_t width)
    : mType(type),
      mWeightDev(weight_dev),
      mWeightSize(weight_size),
      mWeightWidth(width) {}

LookupTablePluginDynamic::LookupTablePluginDynamic(void const* data,
                                                   size_t length) {
  // Deserialize in the same order as serialization
  deserialize_value(&data, &length, &mType);
  deserialize_value(&data, &length, &mWeightSize);
  deserialize_value(&data, &length, &mWeightWidth);
  char const* d = static_cast<char const*>(data);
  cudaMalloc(&mWeightDev, mWeightSize * sizeof(mType));
  cudaMemcpy(
      mWeightDev, d, mWeightSize * sizeof(mType), cudaMemcpyHostToDevice);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* LookupTablePluginDynamic::clone()
    const noexcept {
  auto p = new LookupTablePluginDynamic(
      mType, mWeightDev, mWeightSize, mWeightWidth);
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

nvinfer1::DimsExprs LookupTablePluginDynamic::getOutputDimensions(
    int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  nvinfer1::DimsExprs ret;
  ret.nbDims = inputs[0].nbDims + 1;
  for (int i = 0; i < inputs[0].nbDims; ++i) {
    ret.d[i] = inputs[0].d[i];
  }
  ret.d[inputs[0].nbDims] = exprBuilder.constant(mWeightWidth);
  return ret;
}

bool LookupTablePluginDynamic::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  nvinfer1::PluginTensorDesc const& desc = inOut[pos];
  if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  if (pos == 0) {
    return desc.type == nvinfer1::DataType::kINT32;
  }
  if (pos == 1) {
    if (mType == nvinfer1::DataType::kFLOAT) {
      return desc.type == nvinfer1::DataType::kFLOAT;
    } else {
      return desc.type == nvinfer1::DataType::kHALF;
    }
  }
}

void LookupTablePluginDynamic::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) noexcept {}

size_t LookupTablePluginDynamic::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
  return 0;
}

int32_t LookupTablePluginDynamic::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  int32_t const batchSize = inputDesc->dims.d[0];
  int32_t S;
  if (inputDesc->dims.nbDims == 1) {
    S = 1;
  } else {
    S = inputDesc->dims.d[1];
  }
  int32_t mWeightHeight = mWeightSize / mWeightWidth;
  int32_t status = STATUS_FAILURE;
  auto const inputIds = static_cast<int32_t const*>(inputs[0]);
  if (mType == nvinfer1::DataType::kFLOAT) {
    auto output = static_cast<float*>(outputs[0]);
    auto const Weight = static_cast<const float*>(mWeightDev);
    status = lookup_table<float>(stream,
                                 static_cast<int32_t>(mWeightWidth),
                                 batchSize,
                                 S,
                                 inputIds,
                                 Weight,
                                 mWeightHeight,
                                 output);
  } else if (mType == nvinfer1::DataType::kHALF) {
    auto output = static_cast<half*>(outputs[0]);
    auto const Weight = static_cast<const half*>(mWeightDev);
    status = lookup_table<half>(stream,
                                static_cast<int32_t>(mWeightWidth),
                                batchSize,
                                S,
                                inputIds,
                                Weight,
                                mWeightHeight,
                                output);
  }
  return status;
}

// IPluginV2Ext Methods
nvinfer1::DataType LookupTablePluginDynamic::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
  if (index == 0) {
    assert(mType == nvinfer1::DataType::kHALF ||
           mType == nvinfer1::DataType::kFLOAT);
    return mType;
  }
}

// IPluginV2 Methods
char const* LookupTablePluginDynamic::getPluginType() const noexcept {
  return LOOKUPTABLEPLUGINNAME;
}

char const* LookupTablePluginDynamic::getPluginVersion() const noexcept {
  return PLUGINVERSION;
}

int32_t LookupTablePluginDynamic::getNbOutputs() const noexcept { return 1; }

int32_t LookupTablePluginDynamic::initialize() noexcept { return 0; }

void LookupTablePluginDynamic::terminate() noexcept { cudaFree(mWeightDev); }

size_t LookupTablePluginDynamic::getSerializationSize() const noexcept {
  size_t const wordSize = getElementSize(mType);
  return sizeof(mType)              //
         + sizeof(mWeightSize)      //
         + sizeof(mWeightWidth)     //
         + wordSize * mWeightSize;  //
}

void LookupTablePluginDynamic::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, mType);
  serialize_value(&buffer, mWeightSize);
  serialize_value(&buffer, mWeightWidth);
  char* d = static_cast<char*>(buffer);
  size_t const wordSize = getElementSize(mType);
  serFromDev(&d, static_cast<char*>(mWeightDev), mWeightSize * wordSize);
}

void LookupTablePluginDynamic::destroy() noexcept {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void LookupTablePluginDynamic::setPluginNamespace(
    char const* libNamespace) noexcept {
  mNamespace = libNamespace;
}

char const* LookupTablePluginDynamic::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

LookupTablePluginDynamicCreator::LookupTablePluginDynamicCreator() {}

char const* LookupTablePluginDynamicCreator::getPluginName() const noexcept {
  return LOOKUPTABLEPLUGINNAME;
}

char const* LookupTablePluginDynamicCreator::getPluginVersion() const noexcept {
  return PLUGINVERSION;
}

nvinfer1::PluginFieldCollection const*
LookupTablePluginDynamicCreator::getFieldNames() noexcept {
  return &mFC;
}

bool initializeFields(nvinfer1::PluginFieldCollection const* fc,
                      nvinfer1::Weights* weight,
                      int32_t& mWeightWidth) {  // NOLINT
  bool output_fp16 = false;
  for (int32_t i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);
    if (field_name.compare("lookup_table_weight") == 0) {
      weight->values = fc->fields[i].data;
      weight->count = fc->fields[i].length;
      weight->type = fieldTypeToDataType(fc->fields[i].type);
    }
    if (field_name.compare("lookup_table_weight_width") == 0) {
      assert(fc->fields[i].type == nvinfer1::PluginFieldType::kINT32);
      mWeightWidth = const_cast<int32_t*>(
          static_cast<int32_t const*>(fc->fields[i].data))[0];  // NOLINT
    }
    if (field_name.compare("output_fp16") == 0) {
      assert(fc->fields[i].type == nvinfer1::PluginFieldType::kINT32);
      output_fp16 = static_cast<int32_t const*>(fc->fields[i].data)[0] != 0;
    }
  }
  return output_fp16;
}

nvinfer1::IPluginV2* LookupTablePluginDynamicCreator::createPlugin(
    char const* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  nvinfer1::Weights weight;
  int32_t mWeightWidth;
  bool output_fp16 = initializeFields(fc, &weight, mWeightWidth);
  nvinfer1::DataType type;
  if (output_fp16) {
    type = nvinfer1::DataType::kHALF;
  } else {
    type = nvinfer1::DataType::kFLOAT;
  }
  WeightsWithOwnership mWeight;
  mWeight.convertAndCopy(weight, type);
  void* cudaMem{nullptr};
  cudaMalloc(&cudaMem, getWeightsSize(mWeight, type));
  cudaMemcpy(cudaMem,
             mWeight.values,
             getWeightsSize(mWeight, type),
             cudaMemcpyHostToDevice);
  LookupTablePluginDynamic* p =
      new LookupTablePluginDynamic(type, cudaMem, mWeight.count, mWeightWidth);
  return p;
}

nvinfer1::IPluginV2* LookupTablePluginDynamicCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept {
  return new LookupTablePluginDynamic(serialData, serialLength);
}

void LookupTablePluginDynamicCreator::setPluginNamespace(
    char const* libNamespace) noexcept {
  mNamespace = libNamespace;
}

char const* LookupTablePluginDynamicCreator::getPluginNamespace()
    const noexcept {
  return mNamespace.c_str();
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
