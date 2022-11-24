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

#pragma once

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
<<<<<<< HEAD
=======
#include "paddle/fluid/platform/enforce.h"
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)

class FusedTokenPrunePluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit FusedTokenPrunePluginDynamic(bool with_fp16,
                                        bool keep_first_token,
                                        bool keep_order,
                                        bool flag_varseqlen)
<<<<<<< HEAD
      : keep_first_token_(keep_first_token),
        keep_order_(keep_order),
        flag_varseqlen_(flag_varseqlen) {
    with_fp16_ = with_fp16;
  }
=======
      : with_fp16_(with_fp16),
        keep_first_token_(keep_first_token),
        keep_order_(keep_order),
        flag_varseqlen_(flag_varseqlen) {}
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  FusedTokenPrunePluginDynamic(void const* serial_data, size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &with_fp16_);
    DeserializeValue(&serial_data, &serial_length, &keep_first_token_);
    DeserializeValue(&serial_data, &serial_length, &keep_order_);
    DeserializeValue(&serial_data, &serial_length, &flag_varseqlen_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
<<<<<<< HEAD
    return new FusedTokenPrunePluginDynamic(
        with_fp16_, keep_first_token_, keep_order_, flag_varseqlen_);
=======
    FusedTokenPrunePluginDynamic* ptr = new FusedTokenPrunePluginDynamic(
        with_fp16_, keep_first_token_, keep_order_, flag_varseqlen_);
    ptr->max_batchs_ = max_batchs_;
    ptr->max_token_length_ = max_token_length_;
    ptr->pruned_token_lengths_ = pruned_token_lengths_;
    ptr->token_index_ = token_index_;
    ptr->padding_scores_ = padding_scores_;
    return ptr;
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "fused_token_prune_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override {
    if (flag_varseqlen_) {
      return 5;
    } else {
      return 2;
    }
  }
  int initialize() TRT_NOEXCEPT override { return 0; }

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(with_fp16_) + SerializedSize(keep_first_token_) +
           SerializedSize(keep_order_) + SerializedSize(flag_varseqlen_);
  }
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, with_fp16_);
    SerializeValue(&buffer, keep_first_token_);
    SerializeValue(&buffer, keep_order_);
    SerializeValue(&buffer, flag_varseqlen_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nb_inputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
<<<<<<< HEAD
                       int nb_outputs) TRT_NOEXCEPT override {}
=======
                       int nb_outputs) TRT_NOEXCEPT override {
    max_batchs_ = in[1].max.d[0];
    max_token_length_ = in[1].max.d[1];
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&pruned_token_lengths_,
                                          (max_batchs_ + 1) * sizeof(int32_t)));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(
        &token_index_, max_batchs_ * max_token_length_ * sizeof(int32_t)));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(
        &padding_scores_, max_batchs_ * max_token_length_ * sizeof(half)));
  }
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nb_outputs) const TRT_NOEXCEPT override;

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
              const nvinfer1::PluginTensorDesc* output_desc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) const
      TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }

 private:
<<<<<<< HEAD
  bool keep_first_token_;
  bool keep_order_;
  bool flag_varseqlen_;
=======
  bool with_fp16_;
  bool keep_first_token_;
  bool keep_order_;
  bool flag_varseqlen_;
  int32_t* pruned_token_lengths_;
  int32_t* token_index_;
  int32_t max_batchs_;
  int32_t max_token_length_;
  half* padding_scores_;
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
};

class FusedTokenPrunePluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  FusedTokenPrunePluginDynamicCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "fused_token_prune_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc)
      TRT_NOEXCEPT override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    auto plugin = new FusedTokenPrunePluginDynamic(serial_data, serial_length);
    return plugin;
  }

  void setPluginNamespace(const char* lib_namespace) TRT_NOEXCEPT override {
    plugin_namespace_ = lib_namespace;
  }

  const char* getPluginNamespace() const TRT_NOEXCEPT override {
    return plugin_namespace_.c_str();
  }

 private:
  std::string plugin_namespace_;
  std::string plugin_name_;
  nvinfer1::PluginFieldCollection field_collection_;
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};
REGISTER_TRT_PLUGIN_V2(FusedTokenPrunePluginDynamicCreator);

#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
