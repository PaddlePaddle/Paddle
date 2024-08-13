// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
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

#include "paddle/fluid/inference/tensorrt/plugin/prompt_tuning_emb_layernorm_varseqlen_plugin.h"
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
char const* PTUNING_EMB_LAYER_NORM_VAR_SEQLEN_VERSION_HFACE{"1"};
char const* PTUNING_EMB_LAYER_NORM_VAR_SEQLEN_NAME{
    "PromptTuningEmbLayerNormVarlenPluginDynamic"};
// Static class fields initialization
nvinfer1::PluginFieldCollection
    TrtPromptTuningEmbLayerNormVarSeqlenPluginBaseCreator::mFC{};
std::vector<nvinfer1::PluginField>
    TrtPromptTuningEmbLayerNormVarSeqlenPluginBaseCreator::mPluginAttributes;

TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::
    TrtPromptTuningEmbLayerNormVarSeqlenPluginBase(
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

TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::
    TrtPromptTuningEmbLayerNormVarSeqlenPluginBase(std::string const& name,
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

TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace::
    TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace(
        std::string const& name,
        nvinfer1::DataType const type,
        nvinfer1::Weights const& beta,
        nvinfer1::Weights const& gamma,
        const std::vector<nvinfer1::Weights>& IdsEmb)
    : TrtPromptTuningEmbLayerNormVarSeqlenPluginBase(
          name, type, beta, gamma, IdsEmb) {}

TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace::
    TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace(std::string const& name,
                                                    void const* data,
                                                    size_t length)
    : TrtPromptTuningEmbLayerNormVarSeqlenPluginBase(name, data, length) {
  TRANSFORMER_DEBUG_MSG(
      "TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace deserialize");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt*
TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace::clone() const noexcept {
  TRANSFORMER_DEBUG_MSG(
      "TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace clone");
  auto p = new TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace(
      mLayerName, mType, mBeta, mGamma, mIdsEmb_);
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

nvinfer1::DimsExprs
TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace::getOutputDimensions(
    int32_t outputIndex,
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  for (int i = 1; i < nbInputs - 2; ++i) {
    assert(inputs[i].nbDims == 1);                 // seq length
    assert(inputs[i].nbDims == inputs[1].nbDims);  // same shape
  }
  assert(inputs[0].nbDims == 1);  // pos_id: B+1
  auto one = exprBuilder.constant(1);
  auto Bplus1 = inputs[0].d[0];  // pos_id
  auto B =
      exprBuilder.operation(nvinfer1::DimensionOperation::kSUB, *Bplus1, *one);
  if (outputIndex == 0) {
    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM,
                                     *inputs[1].d[0],
                                     *B);  // sum of seq length
    ret.d[1] = exprBuilder.constant(mLd);
    ret.d[2] = exprBuilder.constant(1);
    ret.d[3] = exprBuilder.constant(1);
    return ret;
  } else if (outputIndex == 1) {
    // This is a hack: we just report some mask size and rely the plugins to
    // play nicely together.
    //      At runtime, depending on the actual maxSeqlen, the size might be
    //      different.
    int32_t maskSize_ = packedMaskSize384;
    auto maskSize = exprBuilder.constant(maskSize_);
    auto fp16maskSize =
        exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
                              *maskSize,
                              *exprBuilder.constant(2));
    nvinfer1::DimsExprs ret;
    ret.nbDims = 2;
    ret.d[0] = B;
    ret.d[1] = fp16maskSize;
    return ret;
  } else if (outputIndex == 2) {
    nvinfer1::DimsExprs ret;
    ret.nbDims = 1;
    ret.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM,
                                     *inputs[nbInputs - 2].d[1],
                                     *one);  // max seqlen
    return ret;
  } else if (outputIndex == 3) {
    nvinfer1::DimsExprs ret = inputs[nbInputs - 2];  // new mask_id
    ret.d[1] = exprBuilder.operation(
        nvinfer1::DimensionOperation::kSUM, *inputs[nbInputs - 2].d[1], *one);
    return ret;
  } else if (outputIndex == 4) {
    nvinfer1::DimsExprs ret = inputs[0];  // new pos_id
    return ret;
  }
}

bool TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  assert(nbOutputs == 5);
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
      pos < (nbInputs - 2)) {  // other ids: check it's the same as input_ids
    return desc.type == prev.type && desc.dims.nbDims == 1 &&
           desc.dims.d[0] == prev.dims.d[0];
  }
  if (pos == nbInputs - 2) {  // mask id
    return desc.type == mType;
  }
  if (pos == nbInputs - 1) {  // dense vector
    return desc.type == mType;
  }
  // embedded sequence
  if (pos == nbInputs) {
    return desc.type == mType && desc.dims.nbDims == 4 && desc.dims.d[2] == 1 &&
           desc.dims.d[3] == 1;
  }
  // mask(HFace)
  if (pos == nbInputs + 1) {
    return desc.type == mType;
  }
  // max seqlen
  if (pos == nbInputs + 2) {
    return desc.type == mType;
  }
  // new mask_id
  if (pos == nbInputs + 3) {
    return desc.type == mType;
  }
  // new pos_id
  if (pos == nbInputs + 4) {
    return desc.dims.nbDims == 1 && desc.type == nvinfer1::DataType::kINT32;
  }
}

void check_tensors(nvinfer1::DynamicPluginTensorDesc const* inputs,
                   int32_t nbInputs,
                   nvinfer1::DynamicPluginTensorDesc const* outputs,
                   int32_t nbOutputs) noexcept {
  // Validate input arguments
  assert(nbOutputs == 5);
  assert(inputs[0].desc.dims.nbDims == 1);
  assert(inputs[0].desc.type == nvinfer1::DataType::kINT32);
  for (int i = 1; i < nbInputs - 2; ++i) {
    assert(inputs[i].desc.dims.nbDims == 1);
    assert(inputs[i].desc.dims.d[0] == inputs[1].desc.dims.d[0]);
    assert(inputs[i].desc.type == nvinfer1::DataType::kINT32);
  }
  assert(outputs[0].desc.dims.nbDims == 4);
  assert(outputs[0].desc.dims.d[2] == 1);
  assert(outputs[0].desc.dims.d[3] == 1);
}

void TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) noexcept {
  TRANSFORMER_DEBUG_MSG(
      "TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace configurePlugin");
  check_tensors(inputs, nbInputs, outputs, nbOutputs);
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

size_t TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
  return 0;
}

int32_t TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  int32_t batchSize = inputDesc[0].dims.d[0] - 1;
  // read out the maximum sequence length from the dummy input
  int32_t const maxSeqlen = inputDesc[nbLookupTables_].dims.d[1] + 1;
  int32_t S;
  if (maxSeqlen <= 128) {
    S = 128;
  } else if (maxSeqlen <= 192) {
    S = 192;
  } else if (maxSeqlen <= 256) {
    S = 256;
  } else if (maxSeqlen <= 384) {
    S = 384;
  } else if (maxSeqlen <= 512) {
    S = 512;
  } else {
    PADDLE_THROW(common::errors::Fatal("The max seqlenth is 512."));
  }
  const float* beta = mBetaDev.get();
  const float* gamma = mGammaDev.get();

  auto output = static_cast<half*>(outputs[0]);
  auto new_pos_id = static_cast<int32_t*>(outputs[4]);
  return prompt_tuning_emb<half>(stream,
                                 static_cast<int32_t>(mLd),
                                 batchSize,
                                 S,
                                 static_cast<int32_t const*>(inputs[0]),
                                 static_cast<int32_t const*>(inputs[1]),
                                 static_cast<int32_t const*>(inputs[2]),
                                 static_cast<half const*>(inputs[4]),
                                 nbLookupTables_,
                                 beta,
                                 gamma,
                                 static_cast<half const*>(mIdsEmbPtrs[0]),
                                 static_cast<half const*>(mIdsEmbPtrs[1]),
                                 static_cast<half const*>(mIdsEmbPtrs[2]),
                                 mIdsVocabSize[0],
                                 mIdsVocabSize[1],
                                 mIdsVocabSize[2],
                                 output,
                                 new_pos_id);
}

// IPluginV2Ext Methods
nvinfer1::DataType
TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
  assert(mType == nvinfer1::DataType::kHALF);
  if (index == 0) {
    return mType;
  } else if (index == 1) {
    return mType;
  } else if (index == 2) {
    return mType;
  } else if (index == 3) {
    return mType;
  } else if (index == 4) {
    return nvinfer1::DataType::kINT32;
  }
}

// IPluginV2 Methods
char const* TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::getPluginType()
    const noexcept {
  return PTUNING_EMB_LAYER_NORM_VAR_SEQLEN_NAME;
}

char const* TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace::getPluginVersion()
    const noexcept {
  return PTUNING_EMB_LAYER_NORM_VAR_SEQLEN_VERSION_HFACE;
}

int32_t TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::getNbOutputs()
    const noexcept {
  return 5;
}

int32_t TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace::initialize() noexcept {
  TRANSFORMER_DEBUG_MSG(
      "TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace initialize");
  return 0;
}

void TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace::terminate() noexcept {
  TRANSFORMER_DEBUG_MSG(
      "TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace terminate");
}

size_t TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::getSerializationSize()
    const noexcept {
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

void TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::serialize(
    void* buffer) const noexcept {
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

void TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::destroy() noexcept {
  // This gets called when the network containing plugin is destroyed
  mBetaDev.reset(nullptr);
  mGammaDev.reset(nullptr);
  for (size_t i = 0; i < mIdsEmbPtrs.size(); ++i) {
    cudaFree(mIdsEmbPtrs[i]);
  }
  delete this;
}

void TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace::destroy() noexcept {
  TRANSFORMER_DEBUG_MSG(
      "TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace destroy");
  TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::destroy();
}

void TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::setPluginNamespace(
    char const* libNamespace) noexcept {
  mNamespace = libNamespace;
}

char const* TrtPromptTuningEmbLayerNormVarSeqlenPluginBase::getPluginNamespace()
    const noexcept {
  return mNamespace.c_str();
}

TrtPromptTuningEmbLayerNormVarSeqlenPluginBaseCreator::
    TrtPromptTuningEmbLayerNormVarSeqlenPluginBaseCreator() = default;

char const*
TrtPromptTuningEmbLayerNormVarSeqlenPluginBaseCreator::getPluginName()
    const noexcept {
  return PTUNING_EMB_LAYER_NORM_VAR_SEQLEN_NAME;
}

char const*
TrtPromptTuningEmbLayerNormVarSeqlenPluginHFaceCreator::getPluginVersion()
    const noexcept {
  return PTUNING_EMB_LAYER_NORM_VAR_SEQLEN_VERSION_HFACE;
}

nvinfer1::PluginFieldCollection const*
TrtPromptTuningEmbLayerNormVarSeqlenPluginBaseCreator::
    getFieldNames() noexcept {
  return &mFC;
}

bool InitializeFields(nvinfer1::PluginFieldCollection const* fc,
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

nvinfer1::IPluginV2*
TrtPromptTuningEmbLayerNormVarSeqlenPluginHFaceCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept {
  TRANSFORMER_DEBUG_MSG("PromptTuningEmbLayerNormVarSeqlenHFace createPlugin");
  nvinfer1::Weights beta;
  nvinfer1::Weights gamma;
  std::vector<nvinfer1::Weights> IdsEmb;
  bool output_fp16 = InitializeFields(fc, &beta, &gamma, &IdsEmb);
  TRANSFORMER_DEBUG_MSG("Building the Plugin...");
  TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace* p =
      new TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace(
          name,
          output_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
          beta,
          gamma,
          IdsEmb);
  return p;
}

nvinfer1::IPluginV2*
TrtPromptTuningEmbLayerNormVarSeqlenPluginHFaceCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept {
  return new TrtPromptTuningEmbLayerNormVarSeqlenPluginHFace(
      name, serialData, serialLength);
}

void TrtPromptTuningEmbLayerNormVarSeqlenPluginBaseCreator::setPluginNamespace(
    char const* libNamespace) noexcept {
  mNamespace = libNamespace;
}

char const*
TrtPromptTuningEmbLayerNormVarSeqlenPluginBaseCreator::getPluginNamespace()
    const noexcept {
  return mNamespace.c_str();
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
