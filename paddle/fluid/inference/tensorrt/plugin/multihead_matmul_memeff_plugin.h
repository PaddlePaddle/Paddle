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
class MultiheadMatmulMemEffPluginDynamic : public DynamicPluginTensorRT{
    public:
    explicit MultiheadMatmulMemEffPluginDynamic(
      int hidden, int head_number, int head_size, float scale, bool with_fp16)
      : hidden_(hidden),
        head_number_(head_number),
        head_size_(head_size),
        scale_(scale) {
    with_fp16_ = with_fp16;
  }
  MultiheadMatmulMemEffPluginDynamic(void const* serial_data, size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &hidden_);
    DeserializeValue(&serial_data, &serial_length, &head_number_);
    DeserializeValue(&serial_data, &serial_length, &head_size_);
    DeserializeValue(&serial_data, &serial_length, &scale_);
    DeserializeValue(&serial_data, &serial_length, &with_fp16_);
    DeserializeValue(&serial_data, &serial_length, &opt_batchsize_);
    DeserializeValue(&serial_data, &serial_length, &opt_seqlen_);
    DeserializeValue(&serial_data, &serial_length, &max_batchsize_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new MultiheadMatmulMemEffPluginDynamic(
        hidden_, head_number_, head_size_, scale_, with_fp16_);
  }
  const char* getPluginType() const TRT_NOEXCEPT override {
    return "multihead_matmul_memeff_plugin";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override;
  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(hidden_) + SerializedSize(head_number_) +
           SerializedSize(head_size_) + SerializedSize(scale_) +
           SerializedSize(with_fp16_) + SerializedSize(opt_batchsize_) +
           SerializedSize(opt_seqlen_) + SerializedSize(max_batchsize_) + ;
  }
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, hidden_);
    SerializeValue(&buffer, head_number_);
    SerializeValue(&buffer, head_size_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, with_fp16_);
    SerializeValue(&buffer, opt_batch_size_);
      SerializeValue(&buffer, opt_seqlen_);
    SerializeValue(&buffer, max_batchsize_);

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
                       int nb_outputs) TRT_NOEXCEPT override;

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
  bool with_fp16_;
  int opt_batchsize_=0;
  int opt_seqlen_=0;
  int max_batchsize_=0;
  void * cu_seqlen=nullptr;
  FusedMultiHeadFlashAttentionKernel const* mKernels{};
  void allocateSeqlens(int32_t max_batchsize);
  void initializeSeqlens(int32_t b, int32_t s, void* cu_seqlens_d, cudaStream_t stream);
  void init(bool);


};

class MultiheadMatmulMemEffPluginDynamicCreater : public nvinfer1::IPluginCreator{
  public:
    MultiheadMatmulMemEffPluginDynamicCreater(){}
    const char* getPluginName() const TRT_NOEXCEPT override {
        return "multihead_matmul_memeff_plugin";
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
    auto plugin = new MultiheadMatmulMemEffPluginDynamic(serial_data, serial_length);
    plugin->init(true);
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
REGISTER_TRT_PLUGIN_V2(MultiheadMatmulMemEffPluginDynamicCreater);

#endif

}
}
}
}