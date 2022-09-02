// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

//for test wangbojun
#define TRT_FT_WINDOWS_ATTENTION 

#ifdef TRT_FT_WINDOWS_ATTENTION
#include "3rdparty/trt_fused_multihead_attention/qkvToContext.h"
#endif  

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000) 
class QkvToContextPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit QkvToContextPluginDynamic(
      int hidden, int head_number, int head_size, float scale, bool with_fp16, bool has_biasqk_mask)
      : hidden_(hidden),
        head_number_(head_number),
        head_size_(head_size),
        scale_(scale),
        has_biasqk_mask_(has_biasqk_mask) {
    with_fp16_ = with_fp16;
  }
  QkvToContextPluginDynamic(void const* serial_data, size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &hidden_);
    DeserializeValue(&serial_data, &serial_length, &head_number_);
    DeserializeValue(&serial_data, &serial_length, &head_size_);
    DeserializeValue(&serial_data, &serial_length, &scale_);
    DeserializeValue(&serial_data, &serial_length, &with_fp16_);
    DeserializeValue(&serial_data, &serial_length, &has_biasqk_mask_);

  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    auto * ptr = new QkvToContextPluginDynamic(
        hidden_, head_number_, head_size_, scale_, with_fp16_, has_biasqk_mask_);
    ptr->ft_dispatcher_fp16_num_head_=ft_dispatcher_fp16_num_head_;
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "qkv_to_context_plugin";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(hidden_) + SerializedSize(head_number_) +
           SerializedSize(head_size_) + SerializedSize(scale_) +
           SerializedSize(with_fp16_) + SerializedSize(has_biasqk_mask_);  
  }
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, hidden_);
    SerializeValue(&buffer, head_number_);
    SerializeValue(&buffer, head_size_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, with_fp16_);
    SerializeValue(&buffer, has_biasqk_mask_);
  }

  nvinfer1::DimsExprs getOutputDimensions(int output_index,
                                          const nvinfer1::DimsExprs* inputs,
                                          int nb_inputs,
                                          nvinfer1::IExprBuilder& expr_builder)
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nb_inputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nb_outputs) TRT_NOEXCEPT override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nb_outputs) const TRT_NOEXCEPT override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
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
  int hidden_;
  int head_number_;
  int head_size_;
  float scale_;
  bool has_biasqk_mask_=false;
  std::unique_ptr<fastertransformer::MHARunner> ft_dispatcher_fp16_;
  int ft_dispatcher_fp16_num_head_=-1;
};

class QkvToContextPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  QkvToContextPluginDynamicCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "qkv_to_context_plugin";
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
    auto plugin = new QkvToContextPluginDynamic(serial_data, serial_length);
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
REGISTER_TRT_PLUGIN_V2(QkvToContextPluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
