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

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)
template <typename T>
class EmbEltwiseLayernormPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit EmbEltwiseLayernormPluginDynamic(
      std::vector<int> emb_sizes, int bias_size, int scale_size,
      int hidden_size, float eps, bool with_fp16,
      std::vector<float const*> input_embs = {}, float const* bias = nullptr,
      float const* scale = nullptr)
      : emb_sizes_(emb_sizes),
        bias_size_(bias_size),
        scale_size_(scale_size),
        hidden_size_(hidden_size),
        eps_(eps),
        with_fp16_(with_fp16),
        embs_(input_embs),
        bias_(bias),
        scale_(scale) {}

  EmbEltwiseLayernormPluginDynamic(void const* serialData,
                                   size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &emb_sizes_);

    embs_gpu_.resize(emb_sizes_.size());
    embs_.resize(emb_sizes_.size());
    for (int i = 0; i < emb_sizes_.size(); i++) {
      cudaMalloc(&embs_gpu_[i], sizeof(float) * emb_sizes_[i]);
      cudaMemcpy(embs_gpu_[i], serialData, emb_sizes_[i] * sizeof(float),
                 cudaMemcpyHostToDevice);
      reinterpret_cast<char const*&>(serialData) +=
          emb_sizes_[i] * sizeof(float);
      serialLength -= emb_sizes_[i] * sizeof(float);
      embs_[i] = nullptr;
    }
    DeserializeValue(&serialData, &serialLength, &bias_size_);
    DeserializeValue(&serialData, &serialLength, &scale_size_);

    cudaMalloc(&bias_gpu_, sizeof(float) * bias_size_);
    cudaMemcpy(bias_gpu_, serialData, bias_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    bias_ = nullptr;
    reinterpret_cast<char const*&>(serialData) += bias_size_ * sizeof(float);
    serialLength -= bias_size_ * sizeof(float);

    cudaMalloc(&scale_gpu_, sizeof(float) * scale_size_);
    cudaMemcpy(scale_gpu_, serialData, scale_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    scale_ = nullptr;
    reinterpret_cast<char const*&>(serialData) += scale_size_ * sizeof(float);
    serialLength -= scale_size_ * sizeof(float);

    DeserializeValue(&serialData, &serialLength, &hidden_size_);
    DeserializeValue(&serialData, &serialLength, &eps_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new EmbEltwiseLayernormPluginDynamic(
        emb_sizes_, bias_size_, scale_size_, hidden_size_, eps_, with_fp16_,
        embs_, bias_, scale_);
  }

  const char* getPluginType() const override {
    return "fused_embedding_eltwise_layernorm_plugin";
  }
  int getNbOutputs() const override { return 1; }
  int initialize() override;

  size_t getSerializationSize() const override {
    // return sizeof(float*) * embs_.size() + sizeof(float*) * 2 +
    //       sizeof(int) * emb_sizes_.size() + sizeof(int) * 3 + sizeof(float);
    int sum_num = 0;
    sum_num += SerializedSize(emb_sizes_);

    for (int i = 0; i < emb_sizes_.size(); i++) {
      sum_num += emb_sizes_[i] * sizeof(float);
    }

    sum_num += SerializedSize(bias_size_);
    sum_num += SerializedSize(scale_size_);

    sum_num += (bias_size_ + scale_size_) * sizeof(float);
    sum_num += SerializedSize(hidden_size_);
    sum_num += SerializedSize(eps_);
    sum_num += SerializedSize(with_fp16_);

    return sum_num;
  }

  void serialize(void* buffer) const override {
    SerializeValue(&buffer, with_fp16_);
    SerializeValue(&buffer, emb_sizes_);
    for (int i = 0; i < emb_sizes_.size(); i++) {
      SerializeCudaPointer(&buffer, embs_gpu_[i], emb_sizes_[i]);
    }
    SerializeValue(&buffer, bias_size_);
    SerializeValue(&buffer, scale_size_);
    SerializeCudaPointer(&buffer, bias_gpu_, bias_size_);
    SerializeCudaPointer(&buffer, scale_gpu_, scale_size_);
    SerializeValue(&buffer, hidden_size_);
    SerializeValue(&buffer, eps_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;

  void destroy() override { delete this; }

 private:
  std::vector<int> emb_sizes_;
  int bias_size_;
  int scale_size_;
  int hidden_size_;
  float eps_;
  bool with_fp16_;

  std::vector<float const*> embs_;
  float const* bias_;
  float const* scale_;

  // data on devices
  float* bias_gpu_;
  float* scale_gpu_;
  std::vector<float*> embs_gpu_;
};

class EmbEltwiseLayernormPluginV2Creator : public nvinfer1::IPluginCreator {
 public:
  EmbEltwiseLayernormPluginV2Creator() {}
  const char* getPluginName() const override {
    return "fused_embedding_eltwise_layernorm_plugin";
  }

  const char* getPluginVersion() const override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &mFieldCollection;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) override {
    bool with_fp16;
    DeserializeValue(&serialData, &serialLength, &with_fp16);

    if (with_fp16) {
#ifdef SUPPORTS_CUDA_FP16
      return new EmbEltwiseLayernormPluginDynamic<half>(serialData,
                                                        serialLength);
#else
      return new EmbEltwiseLayernormPluginDynamic<float>(serialData,
                                                         serialLength);
#endif

    } else {
      return new EmbEltwiseLayernormPluginDynamic<float>(serialData,
                                                         serialLength);
    }
  }

  void setPluginNamespace(const char* libNamespace) override {
    mNamespace = libNamespace;
  }

  const char* getPluginNamespace() const override { return mNamespace.c_str(); }

 private:
  std::string mNamespace;
  std::string mPluginName;
  nvinfer1::PluginFieldCollection mFieldCollection;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(EmbEltwiseLayernormPluginV2Creator);

#endif
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
