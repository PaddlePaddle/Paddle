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

#pragma once
#include <cuda.h>
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "common/bertCommon.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <typename T>
int32_t embSkipLayerNormHFace(cudaStream_t stream,
                              int32_t ld,
                              int32_t B,
                              int32_t S,
                              int32_t** inputIds,
                              int32_t const nbLookupTables,
                              float const* beta,
                              float const* gamma,
                              T** idsEmb,
                              int32_t*,
                              T* output);

template <typename T>
int32_t embSkipLayerNormMTron(cudaStream_t stream,
                              int32_t ld,
                              int32_t B,
                              int32_t S,
                              int32_t** inputIds,
                              int32_t const nbLookupTables,
                              float const* beta,
                              float const* gamma,
                              T** idsEmb,
                              int32_t*,
                              T* output,
                              T* skip);

class EmbLayerNormVarSeqlenPluginBase : public nvinfer1::IPluginV2DynamicExt {
 public:
  EmbLayerNormVarSeqlenPluginBase(
      std::string const& name,
      nvinfer1::DataType const type,
      nvinfer1::Weights const& beta,
      nvinfer1::Weights const& gamma,
      const std::vector<nvinfer1::Weights>& ids_emb);

  EmbLayerNormVarSeqlenPluginBase(std::string const& name,
                                  void const* data,
                                  size_t length);

  // It doesn't make sense to make EmbLayerNormVarSeqlenPlugin without
  // arguments, so we delete default constructor.
  EmbLayerNormVarSeqlenPluginBase() = delete;

  // IPluginV2DynamicExt Methods
  bool supportsFormatCombination(int32_t pos,
                                 nvinfer1::PluginTensorDesc const* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override;
  size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
                          int32_t nbInputs,
                          nvinfer1::PluginTensorDesc const* outputs,
                          int32_t nbOutputs) const noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(
      int32_t index,
      nvinfer1::DataType const* inputTypes,
      int32_t nbInputs) const noexcept override;

  // IPluginV2 Methods
  char const* getPluginType() const noexcept override;
  int32_t getNbOutputs() const noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  char const* getPluginNamespace() const noexcept override;
  void setPluginNamespace(char const* pluginNamespace) noexcept override;

 protected:
  std::string const mLayerName;
  std::string mNamespace;
  cuda_unique_ptr<float> mGammaDev;
  cuda_unique_ptr<float> mBetaDev;
  std::vector<void*> mIdsEmbDev;
  size_t mLd;  // leading dim = hidden size
  std::vector<int32_t> mIdsVocabSize;
  WeightsWithOwnership mBeta;
  WeightsWithOwnership mGamma;
  nvinfer1::DataType mType;
  std::vector<nvinfer1::Weights> mIdsEmb_;
  int32_t nbLookupTables_ = 0;
};

class EmbLayerNormVarSeqlenPluginHFace
    : public EmbLayerNormVarSeqlenPluginBase {
 public:
  EmbLayerNormVarSeqlenPluginHFace(
      std::string const& name,
      nvinfer1::DataType const type,
      nvinfer1::Weights const& beta,
      nvinfer1::Weights const& gamma,
      const std::vector<nvinfer1::Weights>& ids_emb);

  EmbLayerNormVarSeqlenPluginHFace(std::string const& name,
                                   void const* data,
                                   size_t length);

  // It doesn't make sense to make EmbLayerNormVarSeqlenPlugin without
  // arguments, so we delete default constructor.
  EmbLayerNormVarSeqlenPluginHFace() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int32_t outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int32_t nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
                       int32_t nbInputs,
                       nvinfer1::DynamicPluginTensorDesc const* out,
                       int32_t nbOutputs) noexcept override;
  int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                  nvinfer1::PluginTensorDesc const* outputDesc,
                  void const* const* inputs,
                  void* const* outputs,
                  void* workspace,
                  cudaStream_t stream) noexcept override;
  // IPluginV2 Methods
  int32_t initialize() noexcept override;
  void terminate() noexcept override;
  void destroy() noexcept override;
  char const* getPluginVersion() const noexcept override;
};

class EmbLayerNormVarSeqlenPluginMTron
    : public EmbLayerNormVarSeqlenPluginBase {
 public:
  EmbLayerNormVarSeqlenPluginMTron(
      std::string const& name,
      nvinfer1::DataType const type,
      nvinfer1::Weights const& beta,
      nvinfer1::Weights const& gamma,
      const std::vector<nvinfer1::Weights>& ids_emb);

  EmbLayerNormVarSeqlenPluginMTron(std::string const& name,
                                   void const* data,
                                   size_t length);

  // It doesn't make sense to make EmbLayerNormVarSeqlenPlugin without
  // arguments, so we delete default constructor.
  EmbLayerNormVarSeqlenPluginMTron() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int32_t outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int32_t nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
                       int32_t nbInputs,
                       nvinfer1::DynamicPluginTensorDesc const* out,
                       int32_t nbOutputs) noexcept override;
  int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                  nvinfer1::PluginTensorDesc const* outputDesc,
                  void const* const* inputs,
                  void* const* outputs,
                  void* workspace,
                  cudaStream_t stream) noexcept override;

  // IPluginV2 Methods
  int32_t initialize() noexcept override;
  void terminate() noexcept override;
  void destroy() noexcept override;
  char const* getPluginVersion() const noexcept override;
};

class EmbLayerNormVarSeqlenPluginBaseCreator : public nvinfer1::IPluginCreator {
 public:
  EmbLayerNormVarSeqlenPluginBaseCreator();

  char const* getPluginName() const noexcept override;

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

  void setPluginNamespace(char const* pluginNamespace) noexcept override;

  char const* getPluginNamespace() const noexcept override;

 protected:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};

class EmbLayerNormVarSeqlenPluginHFaceCreator
    : public EmbLayerNormVarSeqlenPluginBaseCreator {
 public:
  nvinfer1::IPluginV2* createPlugin(
      char const* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override;
  char const* getPluginVersion() const noexcept override;
  nvinfer1::IPluginV2* deserializePlugin(char const* name,
                                         void const* serialData,
                                         size_t serialLength) noexcept override;
};

class EmbLayerNormVarSeqlenPluginMTronCreator
    : public EmbLayerNormVarSeqlenPluginBaseCreator {
 public:
  nvinfer1::IPluginV2* createPlugin(
      char const* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override;
  char const* getPluginVersion() const noexcept override;
  nvinfer1::IPluginV2* deserializePlugin(char const* name,
                                         void const* serialData,
                                         size_t serialLength) noexcept override;
};
REGISTER_TRT_PLUGIN_V2(EmbLayerNormVarSeqlenPluginHFaceCreator);
REGISTER_TRT_PLUGIN_V2(EmbLayerNormVarSeqlenPluginMTronCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
