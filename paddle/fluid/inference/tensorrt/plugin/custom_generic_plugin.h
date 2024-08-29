// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <NvInfer.h>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_utils.h"
#include "paddle/fluid/inference/tensorrt/plugin_arg_mapping_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/memory/allocation/cuda_allocator.h"
#include "paddle/phi/core/platform/device_context.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

enum class GenerateCustomGenericPluginDataType {
  PLUGIN_BOOL,
  PLUGIN_UINT8,
  PLUGIN_INT8,
  PLUGIN_INT16,
  PLUGIN_INT32,
  PLUGIN_INT64,
  PLUGIN_FP16,
  PLUGIN_FP32,
  PLUGIN_FP64,
  PLUGIN_BF16,
  PLUGIN_SIZE_T,
  PLUGIN_COMPLEX64,
  PLUGIN_COMPLEX128,
  PLUGIN_OPTIONAL,
};

GenerateCustomGenericPluginDataType
ProtoTypeToGenerateCustomGenericPluginDataType(
    framework::proto::VarType_Type proto_type);

class CustomGenericPlugin : public DynamicPluginTensorRT {
 public:
  struct InputOutPutVarInfo {
    std::vector<GenerateCustomGenericPluginDataType> inputs_data_type;
    std::vector<GenerateCustomGenericPluginDataType> outputs_data_type;
  };

 public:
  CustomGenericPlugin() = default;

  CustomGenericPlugin(const paddle::framework::proto::OpDesc& proto_op_desc,
                      const InputOutPutVarInfo& in_out_info,
                      bool with_fp16_ = false);

  CustomGenericPlugin(
      const paddle::framework::proto::OpDesc& proto_op_desc,
      const std::vector<GenerateCustomGenericPluginDataType>& inputs_data_type,
      const std::vector<GenerateCustomGenericPluginDataType>& outputs_data_type,
      bool with_fp16_ = false);

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  CustomGenericPlugin(void const* serialData, size_t serialLength);

  // IPluginV2 method
  const char* getPluginType() const TRT_NOEXCEPT override {
    return "custom_generic_plugin";
  }

  int getNbOutputs() const TRT_NOEXCEPT override;

  int getNbInputs() const TRT_NOEXCEPT;

  // Initialize the layer for execution.
  int initialize() TRT_NOEXCEPT override;

  // Shutdown the layer. This is called when the engine is destroyed
  void terminate() TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override{};

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    size_t sum = 0;
    sum += SerializedSize(inputs_data_type_);
    sum += SerializedSize(outputs_data_type_);
    sum += SerializedSize(with_fp16_);
    sum += op_meta_data_.size();
    return sum;
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override;

  // The Func in IPluginV2
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override;

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

  bool isFp16Supported() { return with_fp16_; }

 private:
  bool with_fp16_{false};
  std::string op_meta_data_;
  framework::proto::OpDesc proto_op_desc_;
  framework::OpDesc op_desc_;

 private:
  std::vector<paddle::Tensor>* tensor_inputs_{nullptr};
  std::vector<paddle::Tensor>* tensor_outputs_{nullptr};

 private:
  std::vector<GenerateCustomGenericPluginDataType> inputs_data_type_;
  std::vector<GenerateCustomGenericPluginDataType> outputs_data_type_;
};

class CustomGenericPluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "custom_generic_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name,
                                                   const void* serial_data,
                                                   size_t serial_length)
      TRT_NOEXCEPT override {
    return new CustomGenericPlugin(serial_data, serial_length);
  }
};

REGISTER_TRT_PLUGIN_V2(CustomGenericPluginCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
