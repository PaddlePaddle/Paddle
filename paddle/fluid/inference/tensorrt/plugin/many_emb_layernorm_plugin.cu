// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved.
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

#include "paddle/fluid/inference/tensorrt/plugin/many_emb_layernorm_plugin.h"
#include <cuda.h>
#include <cstring>
#include <vector>
#include "NvInfer.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

constexpr size_t threadsPerCta128 = 2 * 2 * 32;
constexpr size_t threadsPerCta256 = 1 * 4 * 32;
constexpr size_t threadsPerCta384 = 1 * 8 * 32;
// The number of xmmas in the M dimension. We use one uint32_t per XMMA in the M
// dimension: (s + 16*warps_m - 1) / (16*warps_m);
constexpr size_t xmmasM128 = 4;
constexpr size_t xmmasM256 = 16;
constexpr size_t xmmasM384 = 24;
// Packed mask size per batch. Layout is XMMAS_M * THREADS_PER_CTA.
constexpr size_t packedMaskSize128 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize256 = xmmasM256 * threadsPerCta256;
constexpr size_t packedMaskSize384 = xmmasM384 * threadsPerCta384;
char const* EMB_LAYER_NORM_VERSION{"1"};
char const* EMB_LAYER_NORM_NAME{"ManyEmbLayerNormPluginDynamic"};
// Static class fields initialization
nvinfer1::PluginFieldCollection EmbLayerNormPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> EmbLayerNormPluginCreator::mPluginAttributes;

EmbLayerNormPlugin::EmbLayerNormPlugin(
    std::string const& name,
    nvinfer1::DataType const type,
    nvinfer1::Weights const& beta,
    nvinfer1::Weights const& gamma,
    const std::vector<nvinfer1::Weights>& IdsEmb)
    : mLayerName(name),
      mLd(beta.count),
      mType(type),
      mIdsEmb_(IdsEmb),
      nbLookupTables_(static_cast<int>(IdsEmb.size())) {
  // Assuming Weights.count is the number of elements and not bytes
  assert(beta.count == gamma.count);
  mBeta.convertAndCopy(beta, nvinfer1::DataType::kFLOAT);
  mGamma.convertAndCopy(gamma, nvinfer1::DataType::kFLOAT);
  copyToDevice(&mGamma, sizeof(float) * mGamma.count, &mGammaDev);
  copyToDevice(&mBeta, sizeof(float) * mBeta.count, &mBetaDev);
  for (size_t i = 0; i < mIdsEmb_.size(); ++i) {
    assert(mIdsEmb_[i].count % mLd == 0);
    mIdsVocabSize.push_back(int32_t(mIdsEmb_[i].count / mLd));
    WeightsWithOwnership tem_weight;
    tem_weight.convertAndCopy(mIdsEmb_[i], mType);
    void* cudaMem{nullptr};
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMalloc(&cudaMem, getWeightsSize(tem_weight, mType)));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(cudaMem,
                                          tem_weight.values,
                                          getWeightsSize(tem_weight, mType),
                                          cudaMemcpyHostToDevice));
    mIdsEmbPtrs.push_back(cudaMem);
  }
}

EmbLayerNormPlugin::EmbLayerNormPlugin(std::string const& name,
                                       void const* data,
                                       size_t length)
    : mLayerName(name),
      mGammaDev(nullptr),
      mBetaDev(nullptr),
      mIdsEmbPtrs{},
      mIdsEmb_{} {
  // Deserialize in the same order as serialization
  deserialize_value(&data, &length, &mType);
  deserialize_value(&data, &length, &mLd);
  deserialize_value(&data, &length, &nbLookupTables_);
  for (int32_t i = 0; i < nbLookupTables_; ++i) {
    int32_t tem;
    deserialize_value(&data, &length, &tem);
    mIdsVocabSize.push_back(tem);
  }
  char const* d = static_cast<char const*>(data);
  mBeta.convertAndCopy(&d, mLd, nvinfer1::DataType::kFLOAT);
  mGamma.convertAndCopy(&d, mLd, nvinfer1::DataType::kFLOAT);
  for (int32_t i = 0; i < nbLookupTables_; ++i) {
    nvinfer1::Weights pre_tem_weight;
    pre_tem_weight.type = mType;
    pre_tem_weight.count = mLd * size_t(mIdsVocabSize[i]);
    const auto nbBytes = mLd * size_t(mIdsVocabSize[i]) * getElementSize(mType);
    auto destBuf = new char[nbBytes];
    pre_tem_weight.values = destBuf;
    std::copy_n(d, nbBytes, destBuf);
    d += nbBytes;
    mIdsEmb_.push_back(pre_tem_weight);
  }
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* EmbLayerNormPlugin::clone() const noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormPlugin clone");
  auto p = new EmbLayerNormPlugin(mLayerName, mType, mBeta, mGamma, mIdsEmb_);
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

nvinfer1::DimsExprs EmbLayerNormPlugin::getOutputDimensions(
    int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  assert(outputIndex == 0);
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = exprBuilder.constant(mLd);
  return ret;
}

bool EmbLayerNormPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  assert(nbOutputs == 1);
  nvinfer1::PluginTensorDesc const& prev = inOut[0];
  nvinfer1::PluginTensorDesc const& desc = inOut[pos];
  if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  if (pos == 0) {
    return desc.type == nvinfer1::DataType::kINT32;
  }
  if (0 < pos && pos < nbInputs) {
    assert(desc.dims.nbDims == prev.dims.nbDims);
    for (int i = 0; i < prev.dims.nbDims; ++i) {
      assert(desc.dims.d[i] == prev.dims.d[i]);
    }
    return desc.type == prev.type;
  }
  if (pos == nbInputs) {  // output
    return desc.type == mType && desc.dims.nbDims == 3 &&
           desc.dims.d[0] == prev.dims.d[0] && desc.dims.d[1] == prev.dims.d[1];
  }
}

void EmbLayerNormPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormPlugin configurePlugin");
  assert(static_cast<size_t>(outputs[0].desc.dims.d[2]) ==
         static_cast<size_t>(mLd));
  int32_t const B = inputs[0].desc.dims.d[0];
  if (B > 0) {
    assert(outputs[0].desc.dims.d[0] == B);
  }
  assert(outputs[0].desc.type == mType);
}

size_t EmbLayerNormPlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
  return 0;
}

int32_t EmbLayerNormPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  int32_t batchSize = inputDesc[0].dims.d[0];
  int32_t const maxSeqlen = inputDesc[0].dims.d[1];
  if (maxSeqlen > 512) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "EmbLayerNormPlugin support maxSeqlen is 512"));
  }
  const float* beta = mBetaDev.get();
  const float* gamma = mGammaDev.get();
  if (mType == nvinfer1::DataType::kFLOAT) {
    auto output = static_cast<float*>(outputs[0]);
    if (nbLookupTables_ == 2) {
      return embSkipLayerNorm_2<float>(
          stream,
          static_cast<int32_t>(mLd),
          batchSize,
          maxSeqlen,
          static_cast<int32_t const*>(inputs[0]),
          static_cast<int32_t const*>(inputs[1]),
          nbLookupTables_,
          beta,
          gamma,
          static_cast<float const*>(mIdsEmbPtrs[0]),
          static_cast<float const*>(mIdsEmbPtrs[1]),
          mIdsVocabSize[0],
          mIdsVocabSize[1],
          output);
    } else if (nbLookupTables_ == 3) {
      return embSkipLayerNorm_3<float>(
          stream,
          static_cast<int32_t>(mLd),
          batchSize,
          maxSeqlen,
          static_cast<int32_t const*>(inputs[0]),
          static_cast<int32_t const*>(inputs[1]),
          static_cast<int32_t const*>(inputs[2]),
          nbLookupTables_,
          beta,
          gamma,
          static_cast<float const*>(mIdsEmbPtrs[0]),
          static_cast<float const*>(mIdsEmbPtrs[1]),
          static_cast<float const*>(mIdsEmbPtrs[2]),
          mIdsVocabSize[0],
          mIdsVocabSize[1],
          mIdsVocabSize[2],
          output);
    } else if (nbLookupTables_ == 4) {
      return embSkipLayerNorm_4<float>(
          stream,
          static_cast<int32_t>(mLd),
          batchSize,
          maxSeqlen,
          static_cast<int32_t const*>(inputs[0]),
          static_cast<int32_t const*>(inputs[1]),
          static_cast<int32_t const*>(inputs[2]),
          static_cast<int32_t const*>(inputs[3]),
          nbLookupTables_,
          beta,
          gamma,
          static_cast<float const*>(mIdsEmbPtrs[0]),
          static_cast<float const*>(mIdsEmbPtrs[1]),
          static_cast<float const*>(mIdsEmbPtrs[2]),
          static_cast<float const*>(mIdsEmbPtrs[3]),
          mIdsVocabSize[0],
          mIdsVocabSize[1],
          mIdsVocabSize[2],
          mIdsVocabSize[3],
          output);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Only support 2,3,4 lookup_tables fused "));
    }
  } else if (mType == nvinfer1::DataType::kHALF) {
    auto output = static_cast<half*>(outputs[0]);
    if (nbLookupTables_ == 2) {
      return embSkipLayerNorm_2<half>(stream,
                                      static_cast<int32_t>(mLd),
                                      batchSize,
                                      maxSeqlen,
                                      static_cast<int32_t const*>(inputs[0]),
                                      static_cast<int32_t const*>(inputs[1]),
                                      nbLookupTables_,
                                      beta,
                                      gamma,
                                      static_cast<half const*>(mIdsEmbPtrs[0]),
                                      static_cast<half const*>(mIdsEmbPtrs[1]),
                                      mIdsVocabSize[0],
                                      mIdsVocabSize[1],
                                      output);
    } else if (nbLookupTables_ == 3) {
      return embSkipLayerNorm_3<half>(stream,
                                      static_cast<int32_t>(mLd),
                                      batchSize,
                                      maxSeqlen,
                                      static_cast<int32_t const*>(inputs[0]),
                                      static_cast<int32_t const*>(inputs[1]),
                                      static_cast<int32_t const*>(inputs[2]),
                                      nbLookupTables_,
                                      beta,
                                      gamma,
                                      static_cast<half const*>(mIdsEmbPtrs[0]),
                                      static_cast<half const*>(mIdsEmbPtrs[1]),
                                      static_cast<half const*>(mIdsEmbPtrs[2]),
                                      mIdsVocabSize[0],
                                      mIdsVocabSize[1],
                                      mIdsVocabSize[2],
                                      output);
    } else if (nbLookupTables_ == 4) {
      return embSkipLayerNorm_4<half>(stream,
                                      static_cast<int32_t>(mLd),
                                      batchSize,
                                      maxSeqlen,
                                      static_cast<int32_t const*>(inputs[0]),
                                      static_cast<int32_t const*>(inputs[1]),
                                      static_cast<int32_t const*>(inputs[2]),
                                      static_cast<int32_t const*>(inputs[3]),
                                      nbLookupTables_,
                                      beta,
                                      gamma,
                                      static_cast<half const*>(mIdsEmbPtrs[0]),
                                      static_cast<half const*>(mIdsEmbPtrs[1]),
                                      static_cast<half const*>(mIdsEmbPtrs[2]),
                                      static_cast<half const*>(mIdsEmbPtrs[3]),
                                      mIdsVocabSize[0],
                                      mIdsVocabSize[1],
                                      mIdsVocabSize[2],
                                      mIdsVocabSize[3],
                                      output);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Only support 2,3,4 lookup_tables fused "));
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Unsupported type error, expected [kHALF,kFLOAT]"));
  }
  return STATUS_SUCCESS;
}

// IPluginV2Ext Methods
nvinfer1::DataType EmbLayerNormPlugin::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
  assert(index == 0);
  assert(mType == nvinfer1::DataType::kHALF ||
         mType == nvinfer1::DataType::kFLOAT);
  return mType;
}

// IPluginV2 Methods
char const* EmbLayerNormPlugin::getPluginType() const noexcept {
  return EMB_LAYER_NORM_NAME;
}

char const* EmbLayerNormPlugin::getPluginVersion() const noexcept {
  return EMB_LAYER_NORM_VERSION;
}

int32_t EmbLayerNormPlugin::getNbOutputs() const noexcept { return 1; }

int32_t EmbLayerNormPlugin::initialize() noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormPlugin initialize");
  return 0;
}

void EmbLayerNormPlugin::terminate() noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormPlugin terminate");
}

size_t EmbLayerNormPlugin::getSerializationSize() const noexcept {
  size_t const wordSize = getElementSize(mType);
  return 2 * sizeof(float) * mLd                            // beta + gamma
         + sizeof(mType)                                    //
         + sizeof(mLd)                                      //
         + mIdsVocabSize.size() * sizeof(mIdsVocabSize[0])  //
         + wordSize * mLd *
               accumulate(
                   mIdsVocabSize.begin(), mIdsVocabSize.end(), 0)  // ids emb
         + sizeof(nbLookupTables_);  // numbers of lookup_table
}

void EmbLayerNormPlugin::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, mType);
  serialize_value(&buffer, mLd);
  serialize_value(&buffer, nbLookupTables_);
  for (size_t i = 0; i < mIdsVocabSize.size(); ++i) {
    serialize_value(&buffer, mIdsVocabSize[i]);
  }
  char* d = static_cast<char*>(buffer);
  size_t const wordSize = getElementSize(mType);
  serFromDev(&d, mBetaDev.get(), mLd);
  serFromDev(&d, mGammaDev.get(), mLd);
  for (size_t i = 0; i < mIdsEmbPtrs.size(); ++i) {
    serFromDev(&d,
               static_cast<char*>(mIdsEmbPtrs[i]),
               mLd * mIdsVocabSize[i] * wordSize);
  }
}

void EmbLayerNormPlugin::destroy() noexcept {
  // This gets called when the network containing plugin is destroyed
  mBetaDev.reset(nullptr);
  mGammaDev.reset(nullptr);
  for (size_t i = 0; i < mIdsEmbPtrs.size(); ++i) {
    cudaFree(mIdsEmbPtrs[i]);
  }
  delete this;
}

void EmbLayerNormPlugin::setPluginNamespace(char const* libNamespace) noexcept {
  mNamespace = libNamespace;
}

char const* EmbLayerNormPlugin::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

EmbLayerNormPluginCreator::EmbLayerNormPluginCreator() = default;

char const* EmbLayerNormPluginCreator::getPluginName() const noexcept {
  return EMB_LAYER_NORM_NAME;
}

char const* EmbLayerNormPluginCreator::getPluginVersion() const noexcept {
  return EMB_LAYER_NORM_VERSION;
}

nvinfer1::PluginFieldCollection const*
EmbLayerNormPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

bool initialize_fields(nvinfer1::PluginFieldCollection const* fc,
                       nvinfer1::Weights* beta,
                       nvinfer1::Weights* gamma,
                       std::vector<nvinfer1::Weights>* IdsEmb) {
  bool output_fp16 = false;
  for (int32_t i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);
    if (field_name.compare("bert_embeddings_layernorm_beta") == 0) {
      TRANSFORMER_DEBUG_MSG("Building bert_embeddings_layernorm_beta...");
      beta->values = fc->fields[i].data;
      beta->count = fc->fields[i].length;
      beta->type = fieldTypeToDataType(fc->fields[i].type);
    }

    if (field_name.compare("bert_embeddings_layernorm_gamma") == 0) {
      TRANSFORMER_DEBUG_MSG("Building bert_embeddings_layernorm_gamma...");
      gamma->values = fc->fields[i].data;
      gamma->count = fc->fields[i].length;
      gamma->type = fieldTypeToDataType(fc->fields[i].type);
    }

    if (field_name.compare("output_fp16") == 0) {
      TRANSFORMER_DEBUG_MSG("Building output_fp16...");
      assert(fc->fields[i].type == nvinfer1::PluginFieldType::kINT32);
      output_fp16 = static_cast<int32_t const*>(fc->fields[i].data)[0] != 0;
    }
    if (field_name.compare("bert_embeddings_word_embeddings_" +
                           std::to_string(i - 3)) == 0) {
      TRANSFORMER_DEBUG_MSG(
          ("bert_embeddings_word_embeddings_" + std::to_string(i - 3)).c_str());
      nvinfer1::Weights tem;
      tem.values = fc->fields[i].data;
      tem.count = fc->fields[i].length;
      tem.type = fieldTypeToDataType(fc->fields[i].type);
      IdsEmb->push_back(tem);
    }
  }
  return output_fp16;
}

nvinfer1::IPluginV2* EmbLayerNormPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVar createPlugin");
  nvinfer1::Weights beta;
  nvinfer1::Weights gamma;
  std::vector<nvinfer1::Weights> IdsEmb;
  bool output_fp16 = initialize_fields(fc, &beta, &gamma, &IdsEmb);
  TRANSFORMER_DEBUG_MSG("Building the Plugin...");
  EmbLayerNormPlugin* p = new EmbLayerNormPlugin(
      name,
      output_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
      beta,
      gamma,
      IdsEmb);
  return p;
}

nvinfer1::IPluginV2* EmbLayerNormPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept {
  return new EmbLayerNormPlugin(name, serialData, serialLength);
}

void EmbLayerNormPluginCreator::setPluginNamespace(
    char const* libNamespace) noexcept {
  mNamespace = libNamespace;
}

char const* EmbLayerNormPluginCreator::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
