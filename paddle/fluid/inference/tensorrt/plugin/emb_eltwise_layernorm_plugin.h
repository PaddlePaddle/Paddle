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
class EmbEltwiseLayernormPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit EmbEltwiseLayernormPluginDynamic(float* word_emb, float* pos_emb,
                                            float* sent_emb, float* bias,
                                            float* scale, int64_t word_emb_size,
                                            int64_t pos_emb_size,
                                            int64_t sent_emb_size,
                                            int bias_size, int scale_size,
                                            int hidden_size, float eps)
      : word_emb_(word_emb),
        pos_emb_(pos_emb),
        sent_emb_(sent_emb),
        bias_(bias),
        scale_(scale),
        word_emb_size_(word_emb_size),
        pos_emb_size_(pos_emb_size),
        sent_emb_size_(sent_emb_size),
        bias_size_(bias_size),
        scale_size_(scale_size),
        hidden_size_(hidden_size),
        eps_(eps) {}

  EmbEltwiseLayernormPluginDynamic(void const* serialData,
                                   size_t serialLength) {
    // deserializeBase(serialData, serialLength);
    // DeserializeValue(&serialData, &serialLength, &beta_);
  }
  ~EmbEltwiseLayernormPluginDynamic() {}
  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new EmbEltwiseLayernormPluginDynamic(
        word_emb_, pos_emb_, sent_emb_, bias_, scale_, word_emb_size_,
        pos_emb_size_, sent_emb_size_, bias_size_, scale_size_, hidden_size_,
        eps_);
  }

  const char* getPluginType() const override {
    return "fused_embedding_eltwise_layernorm_plugin";
  }
  int getNbOutputs() const override { return 1; }
  int initialize() override;

  size_t getSerializationSize() const override;
  void serialize(void* buffer) const override;

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
  float* word_emb_;
  float* pos_emb_;
  float* sent_emb_;
  float* bias_;
  float* scale_;

  // data on devices
  float* word_emb_gpu_;
  float* pos_emb_gpu_;
  float* sent_emb_gpu_;
  float* bias_gpu_;
  float* scale_gpu_;

  int64_t word_emb_size_;
  int64_t pos_emb_size_;
  int64_t sent_emb_size_;
  int bias_size_;
  int scale_size_;
  int hidden_size_;
  float eps_;
};
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
