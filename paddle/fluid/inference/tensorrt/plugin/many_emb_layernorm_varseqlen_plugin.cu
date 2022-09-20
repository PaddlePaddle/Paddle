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

#include "paddle/fluid/inference/tensorrt/plugin/many_emb_layernorm_varseqlen_plugin.h"
#include <cuda.h>
#include <cstring>
#include <vector>
#include "NvInfer.h"
#include "common/serialize.h"

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
char const* EMB_LAYER_NORM_VAR_SEQLEN_VERSION_HFACE{"2"};
char const* EMB_LAYER_NORM_VAR_SEQLEN_VERSION_MTRON{"3"};
char const* EMB_LAYER_NORM_VAR_SEQLEN_NAME{"ManyEmbLayerNormPluginDynamic"};
// Static class fields initialization
nvinfer1::PluginFieldCollection EmbLayerNormVarSeqlenPluginBaseCreator::mFC{};
std::vector<nvinfer1::PluginField>
    EmbLayerNormVarSeqlenPluginBaseCreator::mPluginAttributes;

EmbLayerNormVarSeqlenPluginBase::EmbLayerNormVarSeqlenPluginBase(
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
  copyToDevice(mGamma, sizeof(float) * mGamma.count, mGammaDev);
  copyToDevice(mBeta, sizeof(float) * mBeta.count, mBetaDev);
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
    mIdsEmbDev.push_back(cudaMem);
  }
}

EmbLayerNormVarSeqlenPluginBase::EmbLayerNormVarSeqlenPluginBase(
    std::string const& name, void const* data, size_t length)
    : mLayerName(name),
      mGammaDev(nullptr),
      mBetaDev(nullptr),
      mIdsEmbDev{},
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
  mBeta.convertAndCopy(d, mLd, nvinfer1::DataType::kFLOAT);
  mGamma.convertAndCopy(d, mLd, nvinfer1::DataType::kFLOAT);
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

EmbLayerNormVarSeqlenPluginHFace::EmbLayerNormVarSeqlenPluginHFace(
    std::string const& name,
    nvinfer1::DataType const type,
    nvinfer1::Weights const& beta,
    nvinfer1::Weights const& gamma,
    const std::vector<nvinfer1::Weights>& IdsEmb)
    : EmbLayerNormVarSeqlenPluginBase(name, type, beta, gamma, IdsEmb) {}

EmbLayerNormVarSeqlenPluginHFace::EmbLayerNormVarSeqlenPluginHFace(
    std::string const& name, void const* data, size_t length)
    : EmbLayerNormVarSeqlenPluginBase(name, data, length) {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenPluginHFace deserialize");
}

EmbLayerNormVarSeqlenPluginMTron::EmbLayerNormVarSeqlenPluginMTron(
    std::string const& name,
    nvinfer1::DataType const type,
    nvinfer1::Weights const& beta,
    nvinfer1::Weights const& gamma,
    const std::vector<nvinfer1::Weights>& IdsEmb)
    : EmbLayerNormVarSeqlenPluginBase(name, type, beta, gamma, IdsEmb) {}

EmbLayerNormVarSeqlenPluginMTron::EmbLayerNormVarSeqlenPluginMTron(
    std::string const& name, void const* data, size_t length)
    : EmbLayerNormVarSeqlenPluginBase(name, data, length) {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenPluginMTron deserialize");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* EmbLayerNormVarSeqlenPluginHFace::clone()
    const noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenPluginMTron clone");
  auto p = new EmbLayerNormVarSeqlenPluginMTron(
      mLayerName, mType, mBeta, mGamma, mIdsEmb_);
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

nvinfer1::IPluginV2DynamicExt* EmbLayerNormVarSeqlenPluginMTron::clone()
    const noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenPluginMTron clone");
  auto p = new EmbLayerNormVarSeqlenPluginMTron(
      mLayerName, mType, mBeta, mGamma, mIdsEmb_);
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

nvinfer1::DimsExprs EmbLayerNormVarSeqlenPluginHFace::getOutputDimensions(
    int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  for (int i = 1; i < nbInputs - 1; ++i) {
    assert(inputs[i].nbDims == 1);                 // seq length
    assert(inputs[i].nbDims == inputs[1].nbDims);  // same shape
  }
  assert(inputs[0].nbDims == 1);  // pos_id: B+1
  assert(outputIndex == 0 || outputIndex == 1);
  if (outputIndex == 0) {
    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = inputs[1].d[0];  // sum of seq length
    ret.d[1] = exprBuilder.constant(mLd);
    ret.d[2] = exprBuilder.constant(1);
    ret.d[3] = exprBuilder.constant(1);
    return ret;
  }

  // This is a hack: we just report some mask size and rely the plugins to play
  // nicely together.
  //      At runtime, depending on the actual maxSeqlen, the size might be
  //      different.
  int32_t maskSize_ = packedMaskSize384;
  auto maskSize = exprBuilder.constant(maskSize_);
  auto fp16maskSize = exprBuilder.operation(
      nvinfer1::DimensionOperation::kPROD, *maskSize, *exprBuilder.constant(2));
  auto Bplus1 = inputs[0].d[0];  // pos_id
  auto one = exprBuilder.constant(1);
  auto B =
      exprBuilder.operation(nvinfer1::DimensionOperation::kSUB, *Bplus1, *one);
  nvinfer1::DimsExprs ret;
  ret.nbDims = 2;
  ret.d[0] = B;
  ret.d[1] = fp16maskSize;
  return ret;
}

nvinfer1::DimsExprs EmbLayerNormVarSeqlenPluginMTron::getOutputDimensions(
    int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  // Input should be input ids and token ids and cumulative seqlens
  // Output should be the embeddings tensor and mask indices
  for (int i = 1; i < nbInputs - 1; ++i) {
    assert(inputs[i].nbDims == 1);                 // seq length
    assert(inputs[i].nbDims == inputs[1].nbDims);  // same shape
  }
  assert(inputs[0].nbDims == 1);  // pos_id: B+1
  assert(outputIndex == 0 || outputIndex == 1);
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[1].d[0];
  ret.d[1] = exprBuilder.constant(mLd);
  ret.d[2] = exprBuilder.constant(1);
  ret.d[3] = exprBuilder.constant(1);
  return ret;
}

bool EmbLayerNormVarSeqlenPluginBase::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  assert(nbOutputs == 2);
  nvinfer1::PluginTensorDesc const& desc = inOut[pos];
  if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  if (pos == 0) {  // pos_id
    return desc.dims.nbDims == 1 && desc.type == nvinfer1::DataType::kINT32;
  }
  if (pos == 1) {  //  input_id
    return desc.dims.nbDims == 1 && desc.type == nvinfer1::DataType::kINT32;
  }
  nvinfer1::PluginTensorDesc const& prev = inOut[1];  // input_ids
  if (1 < pos &&
      pos < (nbInputs - 1)) {  // other ids: check it's the same as input_ids
    return desc.type == prev.type && desc.dims.nbDims == 1 &&
           desc.dims.d[0] == prev.dims.d[0];
  }
  if (pos == nbInputs - 1) {  // max seq length
    return desc.dims.nbDims == 1;
  }
  // embedded sequence
  if (pos == nbInputs) {
    return desc.type == mType && desc.dims.nbDims == 4 &&
           desc.dims.d[0] == inOut[1].dims.d[0] && desc.dims.d[2] == 1 &&
           desc.dims.d[3] == 1;
  }
  // mask
  return desc.type == nvinfer1::DataType::kHALF;
}

void checkConfigurationInputs(nvinfer1::DynamicPluginTensorDesc const* inputs,
                              int32_t nbInputs,
                              nvinfer1::DynamicPluginTensorDesc const* outputs,
                              int32_t nbOutputs) noexcept {
  // Validate input arguments
  // assert(nbInputs == 4);
  assert(nbOutputs == 2);
  assert(inputs[0].desc.dims.nbDims == 1);
  assert(inputs[0].desc.type == nvinfer1::DataType::kINT32);
  for (int i = 1; i < nbInputs - 1; ++i) {
    assert(inputs[i].desc.dims.nbDims == 1);
    assert(inputs[i].desc.dims.d[0] == inputs[1].desc.dims.d[0]);
    assert(inputs[i].desc.type == nvinfer1::DataType::kINT32);
  }
  assert(outputs[0].desc.dims.nbDims == 4);
  assert(static_cast<size_t>(outputs[0].desc.dims.d[0]) ==
         static_cast<size_t>(inputs[1].desc.dims.d[0]));
  assert(outputs[0].desc.dims.d[2] == 1);
  assert(outputs[0].desc.dims.d[3] == 1);
}

void EmbLayerNormVarSeqlenPluginHFace::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenPluginHFace configurePlugin");
  checkConfigurationInputs(inputs, nbInputs, outputs, nbOutputs);
  assert(static_cast<size_t>(outputs[0].desc.dims.d[1]) ==
         static_cast<size_t>(mLd));
  int32_t const B = inputs[0].desc.dims.d[0] - 1;
  // check mask
  assert(outputs[1].desc.dims.nbDims == 2);
  if (B > 0) {
    assert(outputs[1].desc.dims.d[0] == B);
  }
  assert((outputs[1].desc.dims.d[1] == 2 * packedMaskSize384) ||
         (outputs[1].desc.dims.d[1] == 2 * packedMaskSize128) ||
         (outputs[1].desc.dims.d[1] == 2 * packedMaskSize256));
  assert(outputs[0].desc.type == mType);
  assert(outputs[1].desc.type == nvinfer1::DataType::kHALF);
}

void EmbLayerNormVarSeqlenPluginMTron::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenPluginMTron configurePlugin");
  checkConfigurationInputs(inputs, nbInputs, outputs, nbOutputs);
  assert(static_cast<size_t>(outputs[0].desc.dims.d[1]) ==
         static_cast<size_t>(mLd));
  assert(outputs[1].desc.dims.nbDims == 4);
  assert(static_cast<size_t>(outputs[1].desc.dims.d[0]) ==
         static_cast<size_t>(inputs[1].desc.dims.d[0]));
  assert(static_cast<size_t>(outputs[1].desc.dims.d[1]) ==
         static_cast<size_t>(mLd));
  assert(outputs[1].desc.dims.d[2] == 1);
  assert(outputs[1].desc.dims.d[3] == 1);

  assert(outputs[0].desc.type == mType);
  assert(outputs[1].desc.type == mType);
}

size_t EmbLayerNormVarSeqlenPluginBase::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
  return 0;
}

int32_t EmbLayerNormVarSeqlenPluginHFace::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  int32_t const batchSize = inputDesc[0].dims.d[0] - 1;
  // read out the maximum sequence length from the dummy input
  int32_t const maxSeqlen = inputDesc[nbLookupTables_].dims.d[0];
  int32_t S = 384;
  if (maxSeqlen <= 128) {
    S = 128;
  } else if (maxSeqlen <= 192) {
    S = 192;
  } else if (maxSeqlen <= 256) {
    S = 256;
  }
  const float* beta = mBetaDev.get();
  const float* gamma = mGammaDev.get();
  int32_t** tem_inputs_ptr_dev;
  cudaMalloc(reinterpret_cast<void**>(&tem_inputs_ptr_dev),
             sizeof(void*) * nbLookupTables_);
  cudaMemcpy(tem_inputs_ptr_dev,
             inputs,
             sizeof(void*) * nbLookupTables_,
             cudaMemcpyHostToDevice);
  int32_t* mIdsVocabSize_dev;
  cudaMalloc(reinterpret_cast<void**>(&mIdsVocabSize_dev),
             sizeof(int32_t) * mIdsVocabSize.size());
  cudaMemcpy(mIdsVocabSize_dev,
             &(mIdsVocabSize[0]),
             sizeof(int32_t) * mIdsVocabSize.size(),
             cudaMemcpyHostToDevice);
  if (mType == nvinfer1::DataType::kFLOAT) {
    auto output = static_cast<float*>(outputs[0]);
    float** mIdsEmbDev_float;
    cudaMalloc(reinterpret_cast<void**>(&mIdsEmbDev_float),
               sizeof(void*) * nbLookupTables_);
    cudaMemcpy(mIdsEmbDev_float,
               &(mIdsEmbDev[0]),
               sizeof(void*) * nbLookupTables_,
               cudaMemcpyHostToDevice);
    return embSkipLayerNormHFace<float>(stream,
                                        static_cast<int32_t>(mLd),
                                        batchSize,
                                        S,
                                        tem_inputs_ptr_dev,
                                        nbLookupTables_,
                                        beta,
                                        gamma,
                                        mIdsEmbDev_float,
                                        mIdsVocabSize_dev,
                                        output);
  } else if (mType == nvinfer1::DataType::kHALF) {
    auto output = static_cast<half*>(outputs[0]);
    half** mIdsEmbDev_half;
    cudaMalloc(reinterpret_cast<void**>(&mIdsEmbDev_half),
               sizeof(void*) * nbLookupTables_);
    cudaMemcpy(mIdsEmbDev_half,
               &(mIdsEmbDev[0]),
               sizeof(void*) * nbLookupTables_,
               cudaMemcpyHostToDevice);
    return embSkipLayerNormHFace<half>(stream,
                                       static_cast<int32_t>(mLd),
                                       batchSize,
                                       S,
                                       tem_inputs_ptr_dev,
                                       nbLookupTables_,
                                       beta,
                                       gamma,
                                       mIdsEmbDev_half,
                                       mIdsVocabSize_dev,
                                       output);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unsupported type error, expected [kHALF,kFLOAT]"));
  }
  return STATUS_SUCCESS;
}

int32_t EmbLayerNormVarSeqlenPluginMTron::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  int32_t const batchSize = inputDesc[0].dims.d[0] - 1;
  // read out the maximum sequence length from the dummy input
  int32_t const maxSeqlen = inputDesc[nbLookupTables_].dims.d[0];
  int32_t S = 384;
  if (maxSeqlen <= 128) {
    S = 128;
  } else if (maxSeqlen <= 192) {
    S = 192;
  } else if (maxSeqlen <= 256) {
    S = 256;
  }
  const float* beta = mBetaDev.get();
  const float* gamma = mGammaDev.get();
  int32_t** tem_inputs_ptr_dev;
  cudaMalloc(reinterpret_cast<void**>(&tem_inputs_ptr_dev),
             sizeof(void*) * nbLookupTables_);
  cudaMemcpy(tem_inputs_ptr_dev,
             inputs,
             sizeof(void*) * nbLookupTables_,
             cudaMemcpyHostToDevice);
  int32_t* mIdsVocabSize_dev;
  cudaMalloc(reinterpret_cast<void**>(&mIdsVocabSize_dev),
             sizeof(int32_t) * mIdsVocabSize.size());
  cudaMemcpy(mIdsVocabSize_dev,
             &(mIdsVocabSize[0]),
             sizeof(int32_t) * mIdsVocabSize.size(),
             cudaMemcpyHostToDevice);
  if (mType == nvinfer1::DataType::kFLOAT) {
    auto output = static_cast<float*>(outputs[0]);
    auto skip = static_cast<float*>(outputs[1]);
    float** mIdsEmbDev_float;
    cudaMalloc(reinterpret_cast<void**>(&mIdsEmbDev_float),
               sizeof(void*) * nbLookupTables_);
    cudaMemcpy(mIdsEmbDev_float,
               &(mIdsEmbDev[0]),
               sizeof(void*) * nbLookupTables_,
               cudaMemcpyHostToDevice);
    return embSkipLayerNormMTron<float>(stream,
                                        static_cast<int32_t>(mLd),
                                        batchSize,
                                        S,
                                        tem_inputs_ptr_dev,
                                        nbLookupTables_,
                                        beta,
                                        gamma,
                                        mIdsEmbDev_float,
                                        mIdsVocabSize_dev,
                                        output,
                                        skip);
  } else if (mType == nvinfer1::DataType::kHALF) {
    auto output = static_cast<half*>(outputs[0]);
    auto skip = static_cast<half*>(outputs[1]);
    half** mIdsEmbDev_half;
    cudaMalloc(reinterpret_cast<void**>(&mIdsEmbDev_half),
               sizeof(void*) * nbLookupTables_);
    cudaMemcpy(mIdsEmbDev_half,
               &(mIdsEmbDev[0]),
               sizeof(void*) * nbLookupTables_,
               cudaMemcpyHostToDevice);
    return embSkipLayerNormMTron<half>(stream,
                                       static_cast<int32_t>(mLd),
                                       batchSize,
                                       S,
                                       tem_inputs_ptr_dev,
                                       nbLookupTables_,
                                       beta,
                                       gamma,
                                       mIdsEmbDev_half,
                                       mIdsVocabSize_dev,
                                       output,
                                       skip);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unsupported type error, expected [kHALF,kFLOAT]"));
  }
  return STATUS_SUCCESS;
}

// IPluginV2Ext Methods
nvinfer1::DataType EmbLayerNormVarSeqlenPluginBase::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
  assert(index == 0 || index == 1);
  if (index == 0) {
    assert(mType == nvinfer1::DataType::kHALF ||
           mType == nvinfer1::DataType::kFLOAT);
    return mType;
  }
  return nvinfer1::DataType::kHALF;
}

// IPluginV2 Methods
char const* EmbLayerNormVarSeqlenPluginBase::getPluginType() const noexcept {
  return EMB_LAYER_NORM_VAR_SEQLEN_NAME;
}

char const* EmbLayerNormVarSeqlenPluginHFace::getPluginVersion()
    const noexcept {
  return EMB_LAYER_NORM_VAR_SEQLEN_VERSION_HFACE;
}

char const* EmbLayerNormVarSeqlenPluginMTron::getPluginVersion()
    const noexcept {
  return EMB_LAYER_NORM_VAR_SEQLEN_VERSION_MTRON;
}

int32_t EmbLayerNormVarSeqlenPluginBase::getNbOutputs() const noexcept {
  return 2;
}

int32_t EmbLayerNormVarSeqlenPluginHFace::initialize() noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenPluginHFace initialize");
  return 0;
}

int32_t EmbLayerNormVarSeqlenPluginMTron::initialize() noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenPluginMTron initialize");
  return 0;
}

void EmbLayerNormVarSeqlenPluginHFace::terminate() noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenPluginHFace terminate");
}

void EmbLayerNormVarSeqlenPluginMTron::terminate() noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenPluginMTron terminate");
}

size_t EmbLayerNormVarSeqlenPluginBase::getSerializationSize() const noexcept {
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

void EmbLayerNormVarSeqlenPluginBase::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, mType);
  serialize_value(&buffer, mLd);
  serialize_value(&buffer, nbLookupTables_);
  for (size_t i = 0; i < mIdsVocabSize.size(); ++i) {
    serialize_value(&buffer, mIdsVocabSize[i]);
  }
  char* d = static_cast<char*>(buffer);
  size_t const wordSize = getElementSize(mType);
  serFromDev(d, mBetaDev.get(), mLd);
  serFromDev(d, mGammaDev.get(), mLd);
  for (size_t i = 0; i < mIdsEmbDev.size(); ++i) {
    serFromDev(d,
               static_cast<char*>(mIdsEmbDev[i]),
               mLd * mIdsVocabSize[i] * wordSize);
  }
}

void EmbLayerNormVarSeqlenPluginBase::destroy() noexcept {
  // This gets called when the network containing plugin is destroyed
  mBetaDev.reset(nullptr);
  mGammaDev.reset(nullptr);
  for (size_t i = 0; i < mIdsEmbDev.size(); ++i) {
    cudaFree(mIdsEmbDev[i]);
  }
  delete this;
}

void EmbLayerNormVarSeqlenPluginHFace::destroy() noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenPluginHFace destroy");
  EmbLayerNormVarSeqlenPluginBase::destroy();
}

void EmbLayerNormVarSeqlenPluginMTron::destroy() noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenPluginMTron destroy");
  EmbLayerNormVarSeqlenPluginBase::destroy();
}

void EmbLayerNormVarSeqlenPluginBase::setPluginNamespace(
    char const* libNamespace) noexcept {
  mNamespace = libNamespace;
}

char const* EmbLayerNormVarSeqlenPluginBase::getPluginNamespace()
    const noexcept {
  return mNamespace.c_str();
}

EmbLayerNormVarSeqlenPluginBaseCreator::
    EmbLayerNormVarSeqlenPluginBaseCreator() {}

char const* EmbLayerNormVarSeqlenPluginBaseCreator::getPluginName()
    const noexcept {
  return EMB_LAYER_NORM_VAR_SEQLEN_NAME;
}

char const* EmbLayerNormVarSeqlenPluginHFaceCreator::getPluginVersion()
    const noexcept {
  return EMB_LAYER_NORM_VAR_SEQLEN_VERSION_HFACE;
}

char const* EmbLayerNormVarSeqlenPluginMTronCreator::getPluginVersion()
    const noexcept {
  return EMB_LAYER_NORM_VAR_SEQLEN_VERSION_MTRON;
}

nvinfer1::PluginFieldCollection const*
EmbLayerNormVarSeqlenPluginBaseCreator::getFieldNames() noexcept {
  return &mFC;
}

bool initializeFields(nvinfer1::PluginFieldCollection const* fc,
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

nvinfer1::IPluginV2* EmbLayerNormVarSeqlenPluginHFaceCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenHFace createPlugin");
  nvinfer1::Weights beta;
  nvinfer1::Weights gamma;
  std::vector<nvinfer1::Weights> IdsEmb;
  bool output_fp16 = initializeFields(fc, beta, gamma, IdsEmb);
  TRANSFORMER_DEBUG_MSG("Building the Plugin...");
  EmbLayerNormVarSeqlenPluginHFace* p = new EmbLayerNormVarSeqlenPluginHFace(
      name,
      output_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
      beta,
      gamma,
      IdsEmb);

  return p;
}

nvinfer1::IPluginV2* EmbLayerNormVarSeqlenPluginMTronCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept {
  TRANSFORMER_DEBUG_MSG("EmbLayerNormVarSeqlenMTron createPlugin");
  nvinfer1::Weights beta;
  nvinfer1::Weights gamma;
  std::vector<nvinfer1::Weights> IdsEmb;
  bool output_fp16 = initializeFields(fc, beta, gamma, IdsEmb);
  TRANSFORMER_DEBUG_MSG("Building the Plugin...");
  EmbLayerNormVarSeqlenPluginMTron* p = new EmbLayerNormVarSeqlenPluginMTron(
      name,
      output_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
      beta,
      gamma,
      IdsEmb);
  return p;
}

nvinfer1::IPluginV2* EmbLayerNormVarSeqlenPluginHFaceCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept {
  return new EmbLayerNormVarSeqlenPluginHFace(name, serialData, serialLength);
}

nvinfer1::IPluginV2* EmbLayerNormVarSeqlenPluginMTronCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept {
  return new EmbLayerNormVarSeqlenPluginMTron(name, serialData, serialLength);
}

void EmbLayerNormVarSeqlenPluginBaseCreator::setPluginNamespace(
    char const* libNamespace) noexcept {
  mNamespace = libNamespace;
}

char const* EmbLayerNormVarSeqlenPluginBaseCreator::getPluginNamespace()
    const noexcept {
  return mNamespace.c_str();
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
