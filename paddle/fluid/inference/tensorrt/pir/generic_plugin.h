// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/platform/tensorrt/trt_plugin.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/pir/include/core/operation.h"

namespace paddle::inference::tensorrt::pir {

class SpecialOpConfig {
 public:
  SpecialOpConfig(bool has_format_combination_func,
                  bool has_get_output_data_type_func,
                  bool has_outputs_post_process_func)
      : has_format_combination_func_(has_format_combination_func),
        has_get_output_data_type_func_(has_get_output_data_type_func),
        has_outputs_post_process_func_(has_outputs_post_process_func) {}
  virtual bool supportsFormatCombination(
      int pos,
      const nvinfer1::PluginTensorDesc* in_out,
      int nb_inputs,
      int nb_outputs,
      bool is_fp16_supported) {
    // return a default result
    return false;
  }

  virtual nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* input_types, int nb_inputs) {
    // return a default result
    return input_types[0];
  }

  virtual void outputsPostProcess(
      phi::DeviceContextPool& pool,  // NOLINT
      std::vector<phi::DenseTensor>* dense_tensor_outputs,
      void* const* outputs) {}

  bool HasFormatCombinationFunc() { return has_format_combination_func_; }
  bool HasGetOutputDataTypeFunc() { return has_get_output_data_type_func_; }

  bool HasOutputsPostProcessFunc() { return has_outputs_post_process_func_; }

 protected:
  bool has_format_combination_func_ = false;
  bool has_get_output_data_type_func_ = false;
  bool has_outputs_post_process_func_ = false;
};
class GenericPlugin : public paddle::platform::DynamicPluginTensorRT {
 public:
  GenericPlugin() {}

  GenericPlugin(const std::string& op_name,
                const std::string& attrs_info,
                const std::vector<std::string>& inputs_type_info,
                const std::vector<std::string>& outputs_type_info,
                bool with_fp16 = false);

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  GenericPlugin(void const* serialData, size_t serialLength);

  // IPluginV2 method
  const char* getPluginType() const TRT_NOEXCEPT override {
    return "pir_generic_plugin";
  }
  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
  int getNbOutputs() const TRT_NOEXCEPT override;

  int getNbInputs() const TRT_NOEXCEPT;

  // Initialize the layer for execution.
  int initialize() TRT_NOEXCEPT override;

  // Shutdown the layer. This is called when the engine is destroyed
  void terminate() TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override{};

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    size_t sum = 0;
    sum += paddle::platform::SerializedSize(with_fp16_);
    sum += paddle::platform::SerializedSize(static_cast<int>(op_name_.size()));
    sum += op_name_.size();
    sum += paddle::platform::SerializedSize(
        static_cast<int>(attrs_map_info_.size()));
    sum += attrs_map_info_.size();
    sum += paddle::platform::SerializedSize(
        static_cast<int>(inputs_type_info_.size()));
    for (auto i = 0; i < inputs_type_info_.size(); i++) {
      sum += paddle::platform::SerializedSize(
          static_cast<int>(inputs_type_info_[i].size()));
      sum += inputs_type_info_[i].size();
    }
    sum += paddle::platform::SerializedSize(
        static_cast<int>(outputs_type_info_.size()));
    for (auto i = 0; i < outputs_type_info_.size(); i++) {
      sum += paddle::platform::SerializedSize(
          static_cast<int>(outputs_type_info_[i].size()));
      sum += outputs_type_info_[i].size();
    }
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

  bool isFp16Supported() {
    auto half_dtype = nvinfer1::DataType::kHALF;
    return with_fp16_ &&
           !(phi_kernels_.find(half_dtype) == phi_kernels_.end()) &&
           phi_kernels_[half_dtype]->IsValid();
  }

 private:
  std::string op_name_;
  std::string attrs_map_info_;
  std::vector<std::string> inputs_type_info_;
  std::vector<std::string> outputs_type_info_;
  ::pir::AttributeMap attrs_map_;
  std::vector<::pir::Type> inputs_type_;
  std::vector<::pir::Type> outputs_type_;
  std::unique_ptr<paddle::dialect::OpYamlInfoParser> op_yaml_info_ = nullptr;
  std::unordered_map<std::string, std::unique_ptr<SpecialOpConfig>>
      special_op_config_;

 private:
  std::unordered_map<nvinfer1::DataType, std::unique_ptr<phi::Kernel>>
      phi_kernels_;
  std::unordered_map<nvinfer1::DataType, std::unique_ptr<phi::KernelContext>>
      phi_kernel_contexts_;

  std::vector<phi::DenseTensor>* dense_tensor_inputs_{nullptr};
  std::vector<phi::DenseTensor>* dense_tensor_outputs_{nullptr};
};

class PIRGenericPluginCreator : public paddle::platform::TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "pir_generic_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc)
      TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name,
                                                   const void* serial_data,
                                                   size_t serial_length)
      TRT_NOEXCEPT override {
    return new GenericPlugin(serial_data, serial_length);
  }
};

}  // namespace paddle::inference::tensorrt::pir
